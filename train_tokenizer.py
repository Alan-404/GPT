from preprocessing.tokenizer import Tokenizer
import json
import pandas as pd

ENCODING_FORMAT = 'utf-8'

def train(tokenizer_path: str, data_path: str, token_path: str, saved_tokenizer: str = None):
    assert ".json" in token_path, "Token Path is a JSON file"

    df = pd.read_csv(data_path, sep="\t")
    print(df)
    data = df['input'].to_list() + df['output'].to_list()

    token_item = json.load(open(token_path, encoding=ENCODING_FORMAT))

    tokenizer = Tokenizer(tokenizer_path, special_tokens=token_item['actions'], info_tokens=token_item['dictionary'])

    while(True):
        try:
            max_iteration_input = input("Input max iteration in training tokenizer: ")
            sigma_input = input("Input Sigma: ")

            tokenizer.fit(data=data, max_iterations=int(max_iteration_input) ,sigma=float(sigma_input))

            exit_cmd = input('Do you want to save Tokenizer? (y/n): ').lower().strip()

            if exit_cmd == 'y' or exit_cmd == "yes":
                break

        except Exception as e:
            print(str(e))

    if saved_tokenizer is None:
        saved_tokenizer = tokenizer_path

    tokenizer.save_tokenizer(saved_tokenizer)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--token_path", type=str)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--saved_tokenizer_path", type=str, default=None)

    args = parser.parse_args()

    assert args.tokenizer_path is not None and args.data_path is not None and args.token_path is not None, "Missing Path(s)"

    train(
        tokenizer_path=args.tokenizer_path,
        data_path=args.data_path,
        token_path= args.token_path,
        saved_tokenizer=args.saved_tokenizer_path
    )
    