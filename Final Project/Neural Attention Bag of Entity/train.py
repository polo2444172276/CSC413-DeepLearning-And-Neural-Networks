import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from data import generate_features
from model import NABoE
from optimizer import AdamW
import math

logger = logging.getLogger(__name__)


def train(dataset, embedding, tokenizer, entity_linker, min_count, max_word_length, max_entity_length, batch_size,
          patience, learning_rate, weight_decay, warmup_epochs, dropout_prob, use_gpu, use_word, dim_size):
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    data = generate_features(dataset, tokenizer, entity_linker, min_count, max_word_length, max_entity_length)
    word_vocab = data['word_vocab']
    entity_vocab = data['entity_vocab']

    train_data_loader = DataLoader(data['train'], shuffle=True, batch_size=batch_size)
    dev_data_loader = DataLoader(data['dev'], shuffle=False, batch_size=batch_size)

    dim_size = embedding.syn0.shape[1] #detect dimension from pretrained embeddings
    word_embedding = np.random.uniform(low=-0.05, high=0.05, size=(len(word_vocab), dim_size)) #embedding
    word_embedding[0] = np.zeros(dim_size)
    for word, index in word_vocab.items():
        try:
            word_embedding[index] = embedding.get_word_vector(word)
        except KeyError:
            continue
    entity_embedding = np.random.uniform(low=-0.05, high=0.05, size=(len(entity_vocab), dim_size))
    entity_embedding[0] = np.zeros(dim_size)
    for entity, index in entity_vocab.items():
        try:
            entity_embedding[index] = embedding.get_entity_vector(entity)
        except KeyError:
            continue

    model = NABoE(word_embedding, entity_embedding, len(dataset.label_names), dropout_prob, use_word)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                      warmup=warmup_epochs * len(train_data_loader))
    model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'trainable params:{pytorch_total_params}')

    epoch = 0
    best_val_acc = 0.0
    best_weights = None
    num_epochs_without_improvement = 0
    while True:
        with tqdm(train_data_loader) as pbar:
            model.train()
            for batch in pbar:
                args = {k: v.to(device) for k, v in batch.items() if k != 'label'}
                # args['position_encoding'] = pos_encoding(dim_size, max_entity_length, batch_size)

                # print(args['entity_ids'])
                # print(f"b: {args['entity_ids'].shape}")
                # print(f"b: {args['position_encoding'].shape}")
                logits = model(**args)
                loss = F.cross_entropy(logits, batch['label'].to(device))
                loss.backward()
                optimizer.step()
                model.zero_grad()
                pbar.set_description(f'epoch: {epoch} loss: {loss.item():.8f}')

        epoch += 1
        val_acc = evaluate(model, dev_data_loader, device, 'dev')[0]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.to('cpu').clone() for k, v in model.state_dict().items()}
            num_epochs_without_improvement = 0
        else:
            num_epochs_without_improvement += 1

        if num_epochs_without_improvement >= patience:
            model.load_state_dict(best_weights)
            break

    test_data_loader = DataLoader(data['test'], shuffle=False, batch_size=batch_size)
    return evaluate(model, test_data_loader, device, 'test')


def evaluate(model, data_loader, device, fold):
    model.eval()

    predictions = []
    labels = []
    with torch.no_grad():
        for batch in data_loader:
            args = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            # print(f"b: {args['entity_ids'].shape}")
            # print('abc', model.word_embedding.weight.shape[0])
            # args['position_encoding'] = pos_encoding(model.word_embedding.weight.shape[1], args['entity_ids'].shape[1], args['entity_ids'].shape[0])
            # print(args)
            logits = model(**args)
            predictions += torch.argmax(logits, 1).to('cpu').tolist()
            labels += batch['label'].to('cpu').tolist()

    test_acc = accuracy_score(labels, predictions)
    print(f'accuracy ({fold}): {test_acc:.4f}')

    test_f1 = f1_score(labels, predictions, average='macro')
    print(f'f-measure ({fold}): {test_f1:.4f}')

    return test_acc, test_f1

def pos_encoding(d_model, max_len, batch_size):
    # print('kkk', d_model, max_len, batch_size)
    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model, dtype=torch.float32)
    position = torch.arange(0., max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    # print('a', pe[:, 1::2].shape)
    # print('b', torch.cos(position * div_term).shape)
    # print(max_len)
    pe[:, 1::2] = torch.cos(position * div_term) if max_len % 2 == 0 else torch.cos(position * div_term)[:-1]
    # print('kkkkk', max_len, pe.shape)
    pe = pe.unsqueeze(0)
    # pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)
    return pe
