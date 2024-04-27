def create_config(args):
    from transformers import T5Config, BartConfig

    if 't5' in args.backbone:
        config_class = T5Config
    else:
        return None

    config = config_class.from_pretrained(args.backbone)
    config.dropout_rate = args.dropout
    config.dropout = args.dropout
    config.attention_dropout = args.dropout
    config.activation_dropout = args.dropout
    config.losses = args.losses

    return config


def create_tokenizer(args):
    from transformers import T5Tokenizer, T5TokenizerFast
    from src.tokenization import P5Tokenizer, P5TokenizerFast

    if 'p5' in args.tokenizer:
        tokenizer_class = P5Tokenizer

    tokenizer_name = args.backbone

    tokenizer = tokenizer_class.from_pretrained(
        tokenizer_name,
        max_length=args.max_text_length,
        do_lower_case=args.do_lower_case,
    )

    print(tokenizer_class, tokenizer_name)

    return tokenizer


def create_model(model_class, config=None):
    print(f'Building Model at GPU {args.gpu}')

    model_name = args.backbone

    model = model_class.from_pretrained(
        model_name,
        config=config
    )
    return model


config = create_config(args)

if args.tokenizer is None:
    args.tokenizer = args.backbone

tokenizer = create_tokenizer(args)

model_class = P5Pretraining
model = create_model(model_class, config)

model = model.cuda()

if 'p5' in args.tokenizer:
    model.resize_token_embeddings(tokenizer.vocab_size)

model.tokenizer = tokenizer

# Load Checkpoint
from src.utils import load_state_dict, LossMeter, set_global_logging_level
from pprint import pprint


def load_checkpoint(ckpt_path):
    state_dict = load_state_dict(ckpt_path, 'cpu')
    results = model.load_state_dict(state_dict, strict=False)
    print('Model loaded from ', ckpt_path)
    pprint(results)


ckpt_path = args.load
load_checkpoint(ckpt_path)

import os
from src.all_amazon_templates import all_tasks as task_templates

# data_splits = load_pickle('../data/beauty/rating_splits_augmented.pkl')
data_splits = load_pickle('data/beauty/rating_splits_augmented.pkl')
test_review_data = data_splits['test']

print(len(test_review_data))

print(test_review_data[0])

# data_maps = load_json(os.path.join('../data', 'beauty', 'datamaps.json'))
data_maps = load_json(os.path.join('data', 'beauty', 'datamaps.json'))
print(len(data_maps['user2id'])) # number of users
print(len(data_maps['item2id'])) # number of items