import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
import torchvision


import easydict

class ImageEncoder(nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        self.args = args
        model = torchvision.models.resnet152(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

        pool_func = (
            nn.AdaptiveAvgPool2d
            if args.img_embed_pool_type == "avg"
            else nn.AdaptiveMaxPool2d
        )

        if args.num_image_embeds in [1, 2, 3, 5, 7]:
            self.pool = pool_func((args.num_image_embeds, 1))
        elif args.num_image_embeds == 4:
            self.pool = pool_func((2, 2))
        elif args.num_image_embeds == 6:
            self.pool = pool_func((3, 2))
        elif args.num_image_embeds == 8:
            self.pool = pool_func((4, 2))
        elif args.num_image_embeds == 9:
            self.pool = pool_func((3, 3))

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out  # BxNx2048
    
class ImageBertEmbeddings(nn.Module):
    def __init__(self, args, embeddings):
        super(ImageBertEmbeddings, self).__init__()
        self.args = args
        # Downsamples image
        self.img_embeddings = nn.Linear(args.img_hidden_sz, args.hidden_sz)
        
        # Dont know
        self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.word_embeddings = embeddings.word_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, input_imgs, token_type_ids):
        
        bsz = input_imgs.size(0)
        seq_length = self.args.num_image_embeds + 2  # +2 for CLS and SEP Token

        # Get CLS token ID and embedding
        cls_id = torch.LongTensor([self.args.cls_token_id]).cuda()
        # cls_id = torch.LongTensor([self.args.cls_token_id])
        cls_id = cls_id.unsqueeze(0).expand(bsz, 1)

        cls_token_embeds = self.word_embeddings(cls_id)

        # Get SEP token and embedding
        sep_id = torch.LongTensor([self.args.sep_token_id]).cuda()
        # sep_id = torch.LongTensor([self.args.sep_token_id])
   
        sep_id = sep_id.unsqueeze(0).expand(bsz, 1)
        sep_token_embeds = self.word_embeddings(sep_id)

        # Get downsampled image representation
        imgs_embeddings = self.img_embeddings(input_imgs)

        # Concat cls and img and sep
        token_embeddings = torch.cat(
            [cls_token_embeds, imgs_embeddings, sep_token_embeds], dim=1
        )

        # Get position embeddings
        position_ids = torch.arange(seq_length, dtype=torch.long).cuda()
        # position_ids = torch.arange(seq_length, dtype=torch.long)

        position_ids = position_ids.unsqueeze(0).expand(bsz, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # Get some embeddings
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class MultimodalBertEncoder(nn.Module):
    def __init__(self, args):
        super(MultimodalBertEncoder, self).__init__()
        self.args = args
        bert = AutoModel.from_pretrained(args.bert_model)
        self.txt_embeddings = bert.embeddings

        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)
        self.img_encoder = ImageEncoder(args)
        self.encoder = bert.encoder
        self.pooler = bert.pooler
        # self.clf = nn.Linear(args.hidden_sz, args.n_classes)

    def forward(self, input_txt, attention_mask, segment, input_img):
        bsz = input_txt.size(0)

        if torch.cuda.is_available():
          attention_mask = torch.cat(
              [
                  torch.ones(bsz, self.args.num_image_embeds + 2).long().cuda(),
                  attention_mask,
              ],
              dim=1,
          )
          img_tok = (
            torch.LongTensor(input_txt.size(0), self.args.num_image_embeds + 2)
            .fill_(0)
            .cuda()
        )
        else:
          attention_mask = torch.cat(
              [
                  torch.ones(bsz, self.args.num_image_embeds + 2).long(),
                  attention_mask,
              ],
              dim=1,
          )

          img_tok = (
            torch.LongTensor(input_txt.size(0), self.args.num_image_embeds + 2)
            .fill_(0)
        )
          

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        img = self.img_encoder(input_img)  # BxNx3x224x224 -> BxNx2048
        img_embed_out = self.img_embeddings(img, img_tok)
        txt_embed_out = self.txt_embeddings(input_txt.long(), segment.long())
        encoder_input = torch.cat([img_embed_out, txt_embed_out], 1)  # Bx(TEXT+IMG)xHID

        # encoded_layers = self.encoder(
        #     encoder_input, extended_attention_mask, output_all_encoded_layers=False
        # )
        encoded_layers = self.encoder(
            encoder_input, extended_attention_mask
        )
        return self.pooler(encoded_layers[-1])