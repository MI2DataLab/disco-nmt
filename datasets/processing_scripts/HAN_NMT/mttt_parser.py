import argparse
import os


def create_doc_file(directory, lang, data_type):
    if data_type not in {"train", "dev", "test1"}:
        raise ValueError("You must select one of: 'train', 'dev' or 'test1' data type")

    full_dir = f"{directory}/{lang}/tok"
    file_in = f"ted_{data_type}_{lang}.tok.clean.seekvideo"
    if not os.path.exists(f"{full_dir}/{file_in}"):
        file_in = f"ted_{data_type}_{lang}.tok.seekvideo"
    with open(f"{full_dir}/{file_in}", "r") as f:
        data_seek = f.readlines()

    previous_doc_id = -1
    out_filename = f"ted_{data_type}_{lang}.tok_doc"
    with open(f"{full_dir}/{out_filename}", "w") as f_doc:
        for idx, line in enumerate(data_seek):
            # split line to iterate trough each segment separately, e.g. "<140:1098000> <253:1000> "
            for span in line.split(" "):
                if "<" in span:
                    doc_id = span.split(":")[0][1:]
                    if doc_id != previous_doc_id:
                        previous_doc_id = doc_id
                        if idx == 0:
                            f_doc.write(str(idx))
                        else:
                            f_doc.write(str(idx + 1))
                        f_doc.write("\n")
                        break


def main(input_dir, languages):

    data_types = {"train", "dev", "test1"}

    for lang in languages:
        for data_type in data_types:
            create_doc_file(input_dir, lang, data_type)


if __name__ == "__main__":
    """Script takes as a input directory of MTTT data and list of languages to be parsed
        and generate list of output files ted_{data_type}_{lang_value}.tok_doc where
        data_type = one of train/dev/test1
        lang_value = selected language, e.g. `en-ru` in specific directory
        
        Doc file contain list of beginning and end of each talk, 
        for more details see https://github.com/idiap/HAN_NMT#preprocess
    """

    parser = argparse.ArgumentParser(description='Script for parsing MTTT data into HAN_NMT system')
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--languages', type=str, default="en-ru,en-fr,en-zh,en-cs")
    args = parser.parse_args()

    main(args.input_dir, args.languages.split(","))
