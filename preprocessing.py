from preprocess.data import Tokenizer
import pickle
import json
import io

def train_tokenizer(tokenizer: Tokenizer, dataset: list[str], max_iterations: int, sigma: float):
    tokenizer.fit(data=dataset, max_iterations=max_iterations, sigma=sigma)
    print(f"Dictionary Size: {len(tokenizer.dictionary)}")
    return tokenizer

def text_to_digits(dataset: list[str], tokenizer: Tokenizer, max_length: int = None, start_token: bool = True, end_token: bool = True):
    digits = tokenizer.text_to_sequences(data=dataset, max_length=max_length, start_token=start_token, end_token=end_token)
    return digits

def extract_dataset(dataset_path: str):
    raw_dataset = json.load(open(dataset_path, encoding='utf-8'))
    default_data = io.open("./datasets/vi_qa.txt", encoding='utf-8').read().strip().split('\n')

    dataset = []
    for item in raw_dataset['data']:
        for input in item['inputs']:
            for output in item['outputs']:
                dataset.append(f"{input} <sep> {output}")
    
    tokens = list(raw_dataset['dictionary'].keys()) + raw_dataset['action']

    dataset += default_data
    return dataset, tokens


def program(
        tokenizer_path: str,
        dataset_path: str,
        saved_data_path: str,
        max_length: int,
        start_token: bool,
        end_token: bool
    ):
    dataset, tokens = extract_dataset(dataset_path)
    while True:
        tokenizer = Tokenizer(tokenizer_path, tokens)
        max_iteration_input = input("Input max iteration in training tokenizer: ")
        sigma_input = input("Input Sigma: ")
        tokenizer = train_tokenizer(tokenizer, dataset, int(max_iteration_input), float(sigma_input))
        digits = text_to_digits(dataset, tokenizer, max_length=max_length, start_token=start_token, end_token=end_token)

        print(f"Pre-processed Data Shape: {digits.shape}")

        save = input('Do you want to save data? (y/n): ').lower().strip() == 'y'
        if save:
            tokenizer.save_data(digits, saved_data_path)
            break

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--saved_data_path", type=str)
    parser.add_argument("--max_length", type=int, default=None)

    args = parser.parse_args()


    program(
        tokenizer_path=args.tokenizer_path,
        dataset_path=args.dataset_path,
        saved_data_path=args.saved_data_path,
        max_length=args.max_length,
        start_token=True,
        end_token=True
    )

