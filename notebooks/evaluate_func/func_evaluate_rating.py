# from torch.utils.data import DataLoader, Dataset, Sampler
from src.pretrain_data import get_loader
# from evaluate.utils import rouge_score, bleu_score, unique_sentence_percent, root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity
# from evaluate.metrics4rec import evaluate_all


def evaluate_rating():
    test_task_list = {'rating': ['1-10']  # or '1-6'
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

    gt_ratings = []
    pred_ratings = []
    for i, batch in tqdm(enumerate(zeroshot_test_loader)):
        with torch.no_grad():
            results = model.generate_step(batch)
            gt_ratings.extend(batch['target_text'])
            pred_ratings.extend(results)

    predicted_rating = [(float(r), float(p)) for (r, p) in zip(gt_ratings, pred_ratings) if
                        p in [str(i / 10.0) for i in list(range(10, 50))]]
    RMSE_1 = root_mean_square_error(predicted_rating, 5.0, 1.0)
    print('RMSE {:7.4f}'.format(RMSE_1))
    MAE_1 = mean_absolute_error(predicted_rating, 5.0, 1.0)
    print('MAE {:7.4f}'.format(MAE_1))

    test_task_list = {'rating': ['1-6']  # or '1-10'
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

    gt_ratings = []
    pred_ratings = []
    for i, batch in tqdm(enumerate(zeroshot_test_loader)):
        with torch.no_grad():
            results = model.generate_step(batch)
            gt_ratings.extend(batch['target_text'])
            pred_ratings.extend(results)

    predicted_rating = [(float(r), float(p)) for (r, p) in zip(gt_ratings, pred_ratings) if
                        p in [str(i / 10.0) for i in list(range(10, 50))]]
    RMSE_2 = root_mean_square_error(predicted_rating, 5.0, 1.0)
    print('RMSE {:7.4f}'.format(RMSE_2))
    MAE_2 = mean_absolute_error(predicted_rating, 5.0, 1.0)
    print('MAE {:7.4f}'.format(MAE_2))

    print("evaluate_rating")
    print("rating 1_10")
    print("RMSE_1_10:{} MAE_1_10:{}".format(RMSE_1, MAE_1))
    print("rating 1_6")
    print("RMSE_1_6:{} MAE_1_6:{}".format(RMSE_2, MAE_2))