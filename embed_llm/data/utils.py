import re
from dataclasses import dataclass

RECONSTRUCTION_INSTRUCT = "Text:\n\n||\nReplicate the input text.||Replicated text:\n"

CONTINUATION_INSTRUCT = (
    "Text:\n\n||\nContinue the previous text.||Text continuation:\n..."
)

TEMPLATES_FOR_QA = ["\nQuestion: {question}\nAnswer: "]


@dataclass(frozen=True)
class PromptLabels:
    """Words of the few-shot template. `sep` follows each label."""

    document: str
    question: str
    answer: str
    sep: str

    @property
    def doc_prefix(self) -> str:
        return self.document + self.sep

    @property
    def question_prefix(self) -> str:
        return "\n" + self.question + self.sep

    @property
    def qa_template(self) -> str:
        return self.question_prefix + "{question}\n" + self.answer + self.sep

    @property
    def split_pattern(self) -> str:
        # Separates the documents from the questions and answers of a few-shot prefix
        return "\n\n" + re.escape(self.document) + "|\n" + re.escape(self.question)


PROMPT_LABELS = {
    "en": PromptLabels(document="Document:", question="Question:", answer="Answer:", sep=" "),
    "ja": PromptLabels(document="文書：", question="質問：", answer="答え：", sep=""),
}
