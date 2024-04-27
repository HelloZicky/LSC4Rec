from src.pretrain_data import get_loader
from evaluate.utils import rouge_score, bleu_score, unique_sentence_percent, root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity

test_task_list = {'explanation': ['3-9']  # or '3-12' or '3-3'
                  }
test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

zeroshot_test_loader = get_loader(
    args,
    test_task_list,
    test_sample_numbers,
    split=args.test,
    mode='test',
    batch_size=args.batch_size,
    workers=args.num_workers,
    distributed=args.distributed
)
print(len(zeroshot_test_loader))

tokens_predict = []
tokens_test = []
for i, batch in tqdm(enumerate(zeroshot_test_loader)):
    with torch.no_grad():
        outputs = model.generate(
            batch['input_ids'].to('cuda'),
            min_length=10,
            num_beams=12,
            num_return_sequences=1,
            num_beam_groups=3
        )
        results = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        tokens_predict.extend(results)
        tokens_test.extend(batch['target_text'])

new_tokens_predict = [l.split() for l in tokens_predict]
new_tokens_test = [ll.split() for ll in tokens_test]
BLEU1 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=1, smooth=False)
BLEU4 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=4, smooth=False)
ROUGE = rouge_score(tokens_test, tokens_predict)

print('BLEU-1 {:7.4f}'.format(BLEU1))
print('BLEU-4 {:7.4f}'.format(BLEU4))
for (k, v) in ROUGE.items():
    print('{} {:7.4f}'.format(k, v))

test_task_list = {'explanation': ['3-3']  # or '3-12' or '3-9'
                  }
test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

zeroshot_test_loader = get_loader(
    args,
    test_task_list,
    test_sample_numbers,
    split=args.test,
    mode='test',
    batch_size=args.batch_size,
    workers=args.num_workers,
    distributed=args.distributed
)
print(len(zeroshot_test_loader))

tokens_predict = []
tokens_test = []
for i, batch in tqdm(enumerate(zeroshot_test_loader)):
    with torch.no_grad():
        outputs = model.generate(
            batch['input_ids'].to('cuda'),
            min_length=10
        )
        results = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        tokens_predict.extend(results)
        tokens_test.extend(batch['target_text'])

new_tokens_predict = [l.split() for l in tokens_predict]
new_tokens_test = [ll.split() for ll in tokens_test]
BLEU1 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=1, smooth=False)
BLEU4 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=4, smooth=False)
ROUGE = rouge_score(tokens_test, tokens_predict)

print('BLEU-1 {:7.4f}'.format(BLEU1))
print('BLEU-4 {:7.4f}'.format(BLEU4))
for (k, v) in ROUGE.items():
    print('{} {:7.4f}'.format(k, v))