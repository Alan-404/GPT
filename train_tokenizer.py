from preprocessing.data import Tokenizer
import pandas as pd
import json

ENCODING_FORMAT = 'utf-8'

def train(tokenizer_path: str, manifest_paths: list, token_path: str, limit: int = None, saved_tokenizer: str = None):
    manifest_df = pd.DataFrame()
    print(manifest_paths)
    for path in manifest_paths:
        tmp_df = pd.read_csv(path, sep="\t", encoding=ENCODING_FORMAT)
        manifest_df = pd.concat((manifest_df, tmp_df))
    if limit is not None:
        manifest_df = manifest_df[:limit]

    assert ".json" in token_path, "Token Path is a JSON file"

    token_item = json.load(open(token_path, encoding=ENCODING_FORMAT))

    tokenizer = Tokenizer(tokenizer_path, special_tokens=token_item['actions'], info_tokens=token_item['dictionary'])

    samples = manifest_df['input'].to_list() + manifest_df['output'].to_list()

    while(True):
        try:
            max_iteration_input = input("Input max iteration in training tokenizer: ")
            sigma_input = input("Input Sigma: ")

            tokenizer.fit(data=samples, max_iterations=int(max_iteration_input) ,sigma=float(sigma_input))

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
    parser.add_argument("--manifest_paths", "--names-list", nargs="+", type=str)
    parser.add_argument("--token_path", type=str)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--saved_tokenizer_path", type=str, default=None)

    args = parser.parse_args()

    assert args.tokenizer_path is not None and args.manifest_paths is not None and args.token_path is not None, "Missing Path(s)"

    train(
        tokenizer_path=args.tokenizer_path,
        manifest_paths=args.manifest_paths,
        token_path= args.token_path,
        limit= args.limit,
        saved_tokenizer=args.saved_tokenizer_path
    )
    