from argparse import ArgumentParser

from catalyst.utils import split_dataframe_train_test
from catalyst.utils import (
    create_dataset, create_dataframe, get_dataset_labeling, map_dataframe
)


def create_datasets(
        root: str,
        train_csv: str,
        test_csv: str,
        seed: int = 42
):
    dataset = create_dataset(dirs=f"{root}/*", extension="*.jpg")
    df = create_dataframe(dataset, columns=["class", "filepath"])

    tag_to_label = get_dataset_labeling(df, "class")

    df_with_labels = map_dataframe(
        df,
        tag_column="class",
        class_column="label",
        tag2class=tag_to_label,
        verbose=False
    )

    train_data, valid_data = split_dataframe_train_test(
        df_with_labels, test_size=0.2, random_state=seed)
    train_data.to_csv(train_csv, index=False)
    valid_data.to_csv(test_csv, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--root',
        type=str,
        required=True,
        help='path to root of dataset'
    )
    parser.add_argument(
        '--train_csv',
        type=str,
        required=True,
        help='path to store train csv'
    )
    parser.add_argument(
        '--test_csv',
        type=str,
        required=True,
        help='path to store test csv'
    )
    parser.add_argument(
        '--seed',
        type=int,
        required=False,
        default=42,
        help='seed'
    )

    args = parser.parse_args()
    create_datasets(
        args.root,
        args.train_csv,
        args.test_csv,
        args.seed,
    )
