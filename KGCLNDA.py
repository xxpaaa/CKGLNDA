import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class KGCLNDA(nn.Module):
    """The main model of KGCLNDA"""
    def __init__(self, args, n_circRNAs, n_diseases, n_other_entities, n_relations, A_in=None):
        super(KGCLNDA, self).__init__()
        self.kg_type = args.kg_type
        self.n_circRNAs = n_circRNAs              # circRNA
        self.n_diseases = n_diseases  # disease
        self.n_other_entities = n_other_entities  # other 3 entities, i.e. disease, miRNA, lncRNA
        self.n_relations = n_relations            # 10 relations i.e. circ-disease, circ-miRNA, miRNA-disease, miRNA-lncRNA, lncRNA-disease and their reverse relations

        # For simplicity, we use the same dimention of relations and entities
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim

        self.aggregation_type = args.aggregation_type
        self.conv_dim_list = [args.entity_dim] + eval(args.conv_dim_list)  
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))  

        self.kg_l2loss_lambda = args.kg_l2loss_lambda  
        self.ap_l2loss_lambda = args.ap_l2loss_lambda  

        self.batch_size_ap = args.ap_batch_size
        self.batch_size_kg = args.kg_batch_size

        self.repeat_flag = args.repeat_flag
        self.nei_flag = args.nei_flag

        self.entity_embed = nn.Embedding(self.n_other_entities + self.n_circRNAs, self.entity_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        nn.init.xavier_normal_(self.entity_embed.weight)  
        nn.init.xavier_normal_(self.relation_embed.weight)

        if args.kg_type == 'TF':
            self.kg_dropout = eval(args.kg_dropout)

            kg_dropout_dict = {
                'dr_enc': self.kg_dropout[0],
                'dr_pff': self.kg_dropout[1],
                'dr_sdp': self.kg_dropout[2],
                'dr_mha': self.kg_dropout[3]
            }
            self.encoder = Encoder(args.entity_dim, args.n_layers, args.num_heads, args.d_k, args.d_v,
                                   args.entity_dim, args.d_inner, args.decoder, kg_dropout_dict)
            self.ent_bn = nn.BatchNorm1d(self.entity_dim)
            self.rel_bn = nn.BatchNorm1d(self.entity_dim)

            if args.decoder == 'threemult':
                self.decode_method = 'threemult'
            elif args.decoder == 'twomult':
                self.decode_method = 'twomult'

        if args.kg_type == 'TF':
            self.register_parameter('b', nn.Parameter(torch.zeros(self.n_other_entities + self.n_circRNAs)))
            if args.cl_type == 'CL_CE':
                self.con_loss_lambda = args.con_loss_lambda
                self.supconloss = SupConLoss(temperature=args.tau, contrast_mode="all", base_temperature=args.tau)  # .to(self.A_in.device)
                self.rank_loss = torch.nn.CrossEntropyLoss() #reduction, Default: 'mean'

        # Parameters for relational attention mechanism
        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.entity_dim, self.relation_dim))  
        nn.init.xavier_normal_(self.trans_M)     

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type))

        # Initialize the first attentive matrix to be the laplacian matrix
        self.A_in = nn.Parameter(torch.sparse.FloatTensor(self.n_circRNAs + self.n_other_entities, self.n_circRNAs + self.n_other_entities))
        if A_in is not None:
            self.A_in.data = A_in        # a large laplacian matrix of 10 relations, shape: (n_all_entities, n_all_entities)
        self.A_in.requires_grad = False


    def init_weight(self):
        for m in self.proj:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight.data) #xavier_uniform_
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def calc_ap_embeddings(self):
        ego_embed = self.entity_embed.weight  
        all_embed = [ego_embed]
        
        # Update the node embeddings after the neighbor information is aggregated by 4 gcn layers and store all layes embeddings
        for idx, layer in enumerate(self.aggregator_layers):
            ego_embed = layer(ego_embed, self.A_in)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)  
            all_embed.append(norm_embed)

        all_embed = torch.cat(all_embed, dim=1)  # (n_all_entities, concat_dim = initial_emb+layer1_emb+layer2_emb+layer3emb+layer4_emb)
        return all_embed

    def calc_ap_loss(self, circ_ids, dis_pos_ids, dis_neg_ids): #kg_type
        """
        circ_ids:      (ap_batch_size)
        dis_pos_ids:   (ap_batch_size)
        dis_neg_ids:   (ap_batch_size)
        """
        all_embed = self.calc_ap_embeddings()  # (n_all_entities, concat_dim)
        circ_embed = all_embed[circ_ids]                        # (ap_batch_size, concat_dim)
        dis_pos_embed = all_embed[dis_pos_ids]                  # (ap_batch_size, concat_dim)
        dis_neg_embed = all_embed[dis_neg_ids]     

        pos_score = torch.sum(circ_embed*dis_pos_embed, dim=1)  # (ap_batch_size)
        neg_score = torch.sum(circ_embed*dis_neg_embed, dim=1)   

        ap_loss = F.softplus(neg_score - pos_score)
        # ap_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        ap_loss = torch.mean(ap_loss)

        loss = ap_loss

        return loss

    def calc_kg_loss(self, h, r, pos_t, neg_t, sr_no, kg_type, cl_type):
        """Calculate kg loss on a mini-batch
        Args:
            h(tensor):      (kg_batch_size)
            r(tensor):      (kg_batch_size)
            pos_t(tensor):  (kg_batch_size)
            neg_t(tensor):  (kg_batch_size)
        """
        def rotate(node, edge):
            node_re, node_im = node.chunk(2, dim=-1)
            edge_re, edge_im = edge.chunk(2, dim=-1)
            message_re = node_re * edge_re - node_im * edge_im
            message_im = node_re * edge_im + node_im * edge_re
            message = torch.cat([message_re, message_im], dim=-1)
            return message

        r_embed = self.relation_embed(r)           # (kg_batch_size, relation_dim)
        W_r = self.trans_M[r]                      # (kg_batch_size, entity_dim, relation_dim)

        h_embed = self.entity_embed(h)             # (kg_batch_size, entity_dim)
        pos_t_embed = self.entity_embed(pos_t)     # (kg_batch_size, entity_dim)
        neg_t_embed = self.entity_embed(neg_t)

        # ConvE, RotatE, DistMult, TransE, Transformer
        if kg_type == 'TF' and cl_type == 'CL_CE':
            h_embed = self.ent_bn(h_embed)[:, None]  # (kg_batch_size, 1, entity_dim)
            r_embed = self.rel_bn(r_embed)[:, None]  # (kg_batch_size, 1, entity_dim)
            fusion_feature = torch.cat((h_embed, r_embed), dim=1)  # (kg_batch_size, 2, entity_dim)
            embd_edges = self.encoder(fusion_feature)  # (kg_batch_size, 2, entity_dim)

            if self.decode_method == 'threemult':
                x = embd_edges[:, 0, :] * embd_edges[:, 1, :]
            elif self.decode_method == 'twomult':
                x = embd_edges[:, 1, :]

            cl_x = x

            x = torch.mm(x, self.entity_embed.weight.transpose(1, 0))
            x += self.b.expand_as(x)
            # x = torch.sigmoid(x)
            score = x

            # CELoss
            loss = self.rank_loss(score, pos_t)

            ############ contrastive learning
            tail_emb = F.normalize(pos_t_embed, dim=1)
            x1_node = F.normalize(cl_x, dim=1)
            # calculate SupCon loss
            features = torch.cat((x1_node.unsqueeze(1), tail_emb.unsqueeze(1)), dim=1)
            # SupCon Loss
            supconloss = self.supconloss(features, labels=pos_t, mask=None, mat_ind=torch.cat((sr_no, pos_t)),
                                         repeat_flag=self.repeat_flag, heads=h, nei_flag=self.nei_flag)

            loss = loss + self.con_loss_lambda * supconloss

        return loss

    def _L2_loss_mean(self, x):
        """Compute l2 normalization i.e. (1/2) * ||w||^2 """
        return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)
    
    def update_attention_batch(self, h_list, t_list, r_idx):
        """Update attention matrix for every relation
        Args:
            h_list(list): a id list of head entities appearing in all triples\
            t_list(list): a id list of tail entities appearing in all triples
            r_idx(list) : a id list of relations
        """
        r_embed = self.relation_embed.weight[r_idx]  
        W_r = self.trans_M[r_idx]

        h_embed = self.entity_embed.weight[h_list]
        t_embed = self.entity_embed.weight[t_list]

        r_mul_h = torch.matmul(h_embed, W_r)
        r_mul_t = torch.matmul(t_embed, W_r)
        v_list = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed), dim=1)
        return v_list    # (len(h_list)), )

    def update_attention(self, h_list, t_list, r_list, relations):
        # h_list is a id list of head entity appearing in all triples of 10 relations and so are t_list and r_list
        # relations: id list of 10 relations
        device = self.A_in.device

        rows = []
        cols = []
        values = []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]

            batch_v_list = self.update_attention_batch(batch_h_list, batch_t_list, r_idx)
            rows.append(batch_h_list)    # (len(index_list), )
            cols.append(batch_t_list)    # (len(index_list), )
            values.append(batch_v_list)  # (len(index_list), )

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        indices = torch.stack([rows, cols])
        shape = self.A_in.shape
        A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.A_in.data = A_in.to(device)

    def forward(self, *input, mode): 
        if mode == 'train_ap':
            return self.calc_ap_loss(*input)
        if mode == 'train_kg':
            return self.calc_kg_loss(*input)
        if mode == 'update_att':
            return self.update_attention(*input)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, mat_ind=None, repeat_flag=False, heads=None, nei_flag=False):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # SimCLR loss
            mask = torch.eye(batch_size).float().to(device)
        elif labels is not None:
            # Supconloss
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # concat all contrast features at dim 0
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        if (self.contrast_mode == 'all') and (labels is not None) and (heads is not None):
            if nei_flag:
                heads = heads.contiguous().view(-1, 1)
                object_object_mask = (torch.eq(labels, heads.T) + torch.eq(heads, labels.T)).float().to(device)
                mask[batch_size:, batch_size:] = object_object_mask
            else:
                mask[batch_size:, batch_size:] = torch.zeros((batch_size, batch_size)).float().to(device)

        if (mat_ind is not None) and repeat_flag:
            if mat_ind.shape[0] != batch_size * 2:
                raise ValueError('Num of mat_ind(headRelation_tail) does not match num of features')
            mat_ind = mat_ind.contiguous()
            same_mask = torch.ones((mat_ind.shape[0], mat_ind.shape[0])).float().to(device)
            rep_num_dict = dict()
            for i_ind, num in enumerate(mat_ind.tolist()):
                if num not in rep_num_dict.keys():
                    rep_num_dict[num] = i_ind
                else:
                    same_mask[i_ind, :] = 0
                    same_mask[:, i_ind] = 0

                    if (self.contrast_mode == 'all'):
                        mask[rep_num_dict[num], :] = mask[rep_num_dict[num], :] + mask[i_ind, :]
                    elif (self.contrast_mode == 'one') and (i_ind < batch_size):
                        mask[rep_num_dict[num], :] = mask[rep_num_dict[num], :] + mask[i_ind, :]
                    mask[:, rep_num_dict[num]] = mask[:, rep_num_dict[num]] + mask[:, i_ind]

            mask[mask >= 1.0] = 1.0

        if self.contrast_mode == 'one':
            if repeat_flag:
                same_mask = same_mask[0:batch_size, :]

        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device),  0)

        # mask repeat item
        if repeat_flag:
            # logits_mask = logits_mask * same_mask

            col_vector_same_mask = same_mask[0, :]
            col_vector_same_mask = (col_vector_same_mask != 0).float()
            row_vector_same_mask = same_mask[:, 0]
            row_vector_same_mask = (row_vector_same_mask != 0).float()

            logits_mask = logits_mask[row_vector_same_mask.bool(), :]
            logits_mask = logits_mask[:, col_vector_same_mask.bool()]

            mask = mask[row_vector_same_mask.bool(), :]
            mask = mask[:, col_vector_same_mask.bool()]

            anchor_feature = anchor_feature[row_vector_same_mask.bool(), :]
            contrast_feature = contrast_feature[col_vector_same_mask.bool(), :]

        mask = mask * logits_mask

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob

        # negative samples
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # avoid nan loss when there's one sample for a certain class, e.g., 0,1,...1 for bin-cls , this produce nan for 1st in Batch
        # which also results in batch total loss as nan. such row should be dropped
        pos_per_sample = mask.sum(1)  # B
        pos_per_sample[pos_per_sample < 1e-6] = 1.0
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_sample  # mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(-1, 1).mean()

        return loss


class Encoder(nn.Module): 
    '''Define encoder with n encoder layers'''
    def __init__(self, d_input, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, decoder, dropout_dict):
        super(Encoder, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_input = d_input     
        self.n_layers = n_layers  
        self.n_head = n_head      
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = nn.Dropout(dropout_dict['dr_enc'])

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, decoder, dropout_dict)
            for _ in range(n_layers)])

    def forward(self, edges):    # edges.shape: (batch_size, 2, d_emb) and 2 means the head entity and relation
        enc_output = self.dropout(edges)  

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)

        return enc_output


class EncoderLayer(nn.Module):
    '''The main layer of encoder: multi-head attention + FFN'''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, decoder, dropout_dict):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout_dict)
        self.pos_ffn = PositionwiseFeedForwardUseConv(d_model, d_inner, \
            dropout=dropout_dict['dr_pff'], decode_method=decoder)  

    def forward(self, enc_input):                # enc_input.shape:  (batch_size, 2, d_emb)
        enc_output = self.slf_attn(enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)    # enc_output.shape: (batch_size, 2, d_emb)

        return enc_output


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout_dict):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), dropout=dropout_dict['dr_sdp'])
        self.layer_norm = nn.LayerNorm(d_model)  

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout_dict['dr_mha'])

    def forward(self, q, k, v, mask=None):  # q, k, v shape: (batch_size, 2, d_emb) and 2 means head entity and relation
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # b = a.contiguous() breaks the dependency between these two variables a and b 
        # and here is to make residual=q unchanged
        q = q.transpose(1, 2).contiguous().view(-1, len_q, d_k)  
        k = k.transpose(1, 2).contiguous().view(-1, len_k, d_k)  
        v = v.transpose(1, 2).contiguous().view(-1, len_v, d_v)  

        output = self.attention(q, k, v)    
        output = output.view(sz_b, n_head, len_q, d_v)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)  # b x lq x (h*dv)

        output = self.dropout(self.fc(output))    # transform to dimension d_emb
        output = self.layer_norm(output + residual)

        return output  # output.shape: (batch_size, length_q, d_model)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, dropout=0.2):
        super().__init__()
        self.temperature = temperature      
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):  
        # Input query, key, value are the same size: (batch_size, length_seq, d_emb) and 
        # in this paper, length_seq=2 for a sequence consisting of head entity and relation
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature          # scaled attention scores
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output


class PositionwiseFeedForwardUseConv(nn.Module):
    '''Implement FFN equation in the paper'''
    def __init__(self, d_in, d_hid, dropout=0.3, decode_method='twomult'):
        super(PositionwiseFeedForwardUseConv, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  
        nn.init.kaiming_uniform_(self.w_1.weight, mode='fan_out', nonlinearity='relu')  
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  
        nn.init.kaiming_uniform_(self.w_2.weight, mode='fan_in', nonlinearity='relu')
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)
        self.decode_method = decode_method

    def forward(self, x):   # x.shape: (batch_size, length_seq, dim_embeddings)
        residual = x  
        output = x.transpose(1, 2)  
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output) 
        output = self.layer_norm(output + residual)  # use residual shortcut and layer_norm
        return output


class Aggregator(nn.Module):
    """Aggregate information from neighbors to get the final representation of nodes"""
    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

        if self.aggregator_type == 'gcn':
            self.linear = nn.Linear(self.in_dim, self.out_dim)      
            nn.init.xavier_normal_(self.linear.weight)

        elif self.aggregator_type == 'graphsage':
            self.linear = nn.Linear(self.in_dim * 2, self.out_dim)  
            nn.init.xavier_normal_(self.linear.weight)

        elif self.aggregator_type == 'bi-interaction':
            self.linear1 = nn.Linear(self.in_dim, self.out_dim)     
            self.linear2 = nn.Linear(self.in_dim, self.out_dim)    
            nn.init.xavier_normal_(self.linear1.weight)
            nn.init.xavier_normal_(self.linear2.weight)

        else:
            raise NotImplementedError

    def forward(self, ego_embeddings, A_in):
        """
        ego_embeddings:  (n_all_entities, in_dim)
        A_in:            (n_all_entities, n_all_entities), torch.sparse.FloatTensor
        """
        side_embeddings = torch.matmul(A_in, ego_embeddings)  # A_in is the laplacian matrix which removed its own connection

        if self.aggregator_type == 'gcn':
            embeddings = ego_embeddings + side_embeddings          
            embeddings = self.activation(self.linear(embeddings))  

        elif self.aggregator_type == 'graphsage':
            embeddings = torch.cat([ego_embeddings, side_embeddings], dim=1)
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'bi-interaction':
            sum_embeddings = self.activation(self.linear1(ego_embeddings + side_embeddings))
            bi_embeddings = self.activation(self.linear2(ego_embeddings * side_embeddings))
            embeddings = bi_embeddings + sum_embeddings

        embeddings = self.message_dropout(embeddings) 
        return embeddings    # return updated embeddings after a gcn layer of information aggregation


