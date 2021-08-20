import json
import datasets
from random import randint
from nlp import DatasetInfo, BuilderConfig, SplitGenerator, Split, utils

logger = datasets.logging.get_logger(__name__)


class ProcurementConfig(datasets.BuilderConfig):
    """BuilderConfig for Procurement."""

    def __init__(self, **kwargs):
        """BuilderConfig for Procurement.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ProcurementConfig, self).__init__(**kwargs)


class Procurement(datasets.GeneratorBasedBuilder):
    """Procurement: Version 1.0.0"""

    BUILDER_CONFIGS = [
        ProcurementConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="Procurement dataset",
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

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for article in data:
                schema, question = article["question"].split("</s>")
                question = f"{question} </s> {schema}"
                # print(question, answer)
                yield article["id"], {
                    "question": article["question"],
                    "answer": article["answer"],
                }
