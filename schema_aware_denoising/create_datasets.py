import json
import datasets
from random import choice, randint
from nlp import DatasetInfo, BuilderConfig, SplitGenerator, Split, utils
import ast
import random
import re
from tqdm import tqdm

logger = datasets.logging.get_logger(__name__)


class Text2SQLConfig(datasets.BuilderConfig):
    """BuilderConfig for Text2SQL."""

    def __init__(self, **kwargs):
        """BuilderConfig for Text2SQL.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Text2SQLConfig, self).__init__(**kwargs)


class Text2SQL(datasets.GeneratorBasedBuilder):
    """Text2SQL: Version 1.0.0"""

    BUILDER_CONFIGS = [
        Text2SQLConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="Text2SQL dataset",
            features=datasets.Features(
                {
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="",
            citation="",
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(self.config.data_files)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["validation"]},
            ),
        ]

    def _wikisql_example_erosion(self, example, dataset):
        schema = ""
        sql = example["sql"].replace("table", "` table `")
        types = ast.literal_eval(example["header_types"])
        conds = ast.literal_eval(example["conds"])
        for condition in conds:
            sql = sql.replace(condition, f"` {condition} `")

        ## noising step 1: Repermute the schema
        table_header = ast.literal_eval(example["header"])
        header_types = list(zip(table_header, types))
        random.shuffle(header_types)
        table_header, types = zip(*header_types)
        table_header, types = list(table_header), list(types)

        ## noising step 2: Additional column to schema with probability = 0.3 from train set examples
        p_add = random.random()
        if p_add > 0.7:
            additional_random_example = random.choice(dataset)
            additional_random_example_header = ast.literal_eval(
                additional_random_example["header"]
            )
            additional_column = random.choice(additional_random_example_header)
            additional_column_index = additional_random_example_header.index(
                additional_column
            )
            additional_column_type = ast.literal_eval(
                additional_random_example["header_types"]
            )[additional_column_index]

            table_header.append(additional_column)
            types.append(additional_column_type)

        assert len(table_header) == len(types)
        for index, value in enumerate(table_header):

            col_name = value.strip().replace("'", "")
            col_num = f"<col{index}>"
            col_type = types[index].replace("'", "")
            ## noising step 3: Dropping each column with probability = 0.1
            p_drop = random.random()
            if p_drop <= 0.9:
                schema += f"{col_num} {col_name} : {col_type} "
                sql = sql.replace(col_name, f"` {col_num} `")
            else:
                ## update corresponding sql for dropped column
                sql = sql.replace(col_name, f"` <unk> `")

        schema = schema.strip()
        return schema, example["question"], sql

    def _wikisql_example_shuffle(self, question, answer):
        target_type = "<2ql>"
        query_entities = [
            entity.strip().lower()
            for entity in re.findall("`([^`]*)`", answer)
            if entity != " table "
        ]
        shuffled_entities = [
            entity.strip().lower()
            for entity in re.findall("`([^`]*)`", answer)
            if entity != " table "
        ]
        random.shuffle(shuffled_entities)
        ## noising step 5: swap sql and NL question where sql becomes question and NL sentence becomes answer
        p_swap = random.random()
        if p_swap > 0.5:
            question, answer = answer, question
            target_type = "<2nl>"

        noised_answer = answer.lower()
        for entity, shuffled_entity in zip(query_entities, shuffled_entities):
            noised_answer = noised_answer.replace(f"{entity}", f"{shuffled_entity}")
        question = noised_answer
        return target_type, question, answer

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            count = 0
            for article in tqdm(data):
                target_type = "<2ql>"
                schema, question, answer = self._wikisql_example_erosion(article, data)
                ## noising step 4: Additional column to schema with probability = 0.3 from train set examples
                """p_shuffle = random.random()
                if p_shuffle > 0.7:
                    target_type, question, answer = self._wikisql_example_shuffle(
                        question, answer
                    )"""
                question = f"{target_type} </s> {question} </s> {schema}"
                yield article["id"], {"question": question, "answer": answer}


if __name__ == "__main__":
    pass
