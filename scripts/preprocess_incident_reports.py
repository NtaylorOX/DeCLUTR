#!/usr/bin/env python3
import io
import re
import os
import zipfile
from pathlib import Path
from typing import List, Optional

import requests
import typer
from declutr.common.util import sanitize_text

import argparse
# Emoji's used in typer.secho calls
# See: https://github.com/carpedm20/emoji/blob/master/emoji/unicode_codes.py"
SAVING = "\U0001F4BE"
DOWNLOAD = "\U00002B07"


def _write_output_to_disk(text: List[str], output_filepath: Path) -> None:
    """Writes a list of documents, `text`, to the file `output_filepath`, one document per line."""
    # Create the directory path if it doesn't exist
    output_filepath = Path(output_filepath)
    output_filepath.parents[0].mkdir(parents=True, exist_ok=True)

    with open(output_filepath, "w", encoding = 'utf-8') as f:
        # TODO (John): In the future, it might make sense to both batch and shard:
        # 1) Batch, meaning write batches of documents to a file as opposed to 1 at a time
        # 2) Shard, meaning break a file up into shard_size // len(text) files, and return a
        #    directory instead. Loading a dataset like this is supported in AllenNLP (see:
        #    https://docs.allennlp.org/master/api/data/dataset_readers/sharded_dataset_reader/)
        with typer.progressbar(text, label="Writing to disk") as progress:
            for doc in progress:                
                f.write(doc.strip() + "\n")
    typer.secho(
        f"{SAVING} {len(text)} preprocessed documents saved to: {output_filepath}",
        bold=True,
    )


def main(
    input_filepath: Path,
    output_filepath: Path,
    segment_sentences: bool = False,
    lowercase: bool = False,
    min_length: Optional[int] = None,
    max_instances: Optional[int] = None,
    pretrained_model_name_or_path: Optional[str] = None,
) -> None:
    """Lightly preprocess the text dataset. If `min_length is not None`, only documents
    with at least this many tokens are retained. If `pretrained_model_name_or_path` is not None, the
    tokenizer will be loaded as `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)`
    using the HuggingFace Transformers library. Otherwise `str.split()` is used. This argument has
    no effect if `min-length is None`. If `segment_sentences` is provided, individual sentences
    will be returned instead of documents. You must have the `"en_core_web_sm"` spacy model
    installed to segment sentences.
    """
    # Setup the pre-trained tokenizer, if specified
    if min_length is not None:
        if pretrained_model_name_or_path is not None:
            # Import transformers here to prevent ImportError errors if the
            # user doesn't want to use it.
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path).tokenize
        else:
            tokenizer = lambda x: x.split()  # noqa
    else:
        tokenizer = None

    # Setup spacy lang object if we are segmenting sentences
    if segment_sentences:
        import spacy

        nlp = spacy.load("en_core_web_sm", disable=["ner"])

    # # Download WikiText-103
    # r = requests.get(WIKITEXT_103_URL, stream=True)
    # z = zipfile.ZipFile(io.BytesIO(r.content))
    # partition_filenames = z.namelist()[1:]
    # typer.secho(f"{DOWNLOAD} Downloaded WikiText-103", bold=True)    
 

    preprocessed_documents: List[str] = []
    
    # load in the one big training file
    with open(input_filepath, 'r') as f: 
        text = f.readlines()

       
        # Strip out subtitles and split the text into documents
        documents = text
        
        if segment_sentences:
            documents = (sent.text for doc in documents for sent in nlp(doc).sents)  # type: ignore

        with typer.progressbar(
            documents, length=max_instances, label=typer.style("Preprocessing text", bold=True)
        ) as progress:
            for doc in progress:
                doc = sanitize_text(doc, lowercase=lowercase)
                # print(f"Doc processed is: {doc}")
                if not doc:
                    continue

                # Retain documents if the length of their shortest document is
                # equal to or greater than the minimum specified length
                if tokenizer is not None:
                    num_tokens = len(tokenizer(doc))
                    # print(f"num tokens:{num_tokens}")
                    
                
                    if min_length and num_tokens < min_length:
                        continue

                if max_instances and len(preprocessed_documents) >= max_instances:
                    break
                preprocessed_documents.append(doc)
                progress.update(1)

    _write_output_to_disk(preprocessed_documents, output_filepath)

    # ensure file close
    f.close()

if __name__ == "__main__":
    



    # typer.run(main)
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_filepath",
                        default = "E:/working_incident_data/cleaned_data/training_all_text_100000.txt",
                        type=str,
                        help = "The data path to the file containing the saved model etc")
    
    parser.add_argument("--save_directory",
                        default = "E:/working_incident_data/cleaned_data/declutr_lm/train.txt",
                        type=str,
                        help = "The data path to the file containing the saved model etc")
    parser.add_argument("--segment_sentences",
                        default = False,
                        type=bool,
                        help = "Whether to segment the documents into sentences using spaCy")
    parser.add_argument("--min_length",
                        default = 32,
                        type=int,
                        help = "The minimum number of tokens a document should have to be kept")
    parser.add_argument("--max_length",
                        default = 50,
                        type=int,
                        help = "The max number of tokens a document should have to be kept")  
    parser.add_argument("--max_instances",
                        default = None,
                        type=int,
                        help = "The total maximum of documents to keep")
    parser.add_argument("--pretrained_model_name_or_path",
                        default = None,
                        type=str,
                        help = "The name of HF model")
        
    
    # create args object
    args = parser.parse_args()
    
    # now run
    main(input_filepath=args.input_filepath,
            output_filepath = args.save_directory,
            segment_sentences= args.segment_sentences,
            min_length = args.min_length,
            max_instances= args.max_instances,
            pretrained_model_name_or_path = args.pretrained_model_name_or_path)