import dataclasses
import logging
import sys
from dataclasses import dataclass
from typing import List, Optional, Callable
import click
import coloredlogs
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token as SpacyToken
from bottle import Bottle, request, abort, response
import json
import meinheld


@dataclass
class Token:
    text: str
    tag: str
    lemma: str
    entity: Optional[str] = None

    def serialize(self):
        d = dataclasses.asdict(self)
        if self.entity is None:
            del d['entity']
        return d


class Handler:
    def __init__(self, tagging_nlp: Language, ner_nlp: Optional[Language] = None) -> None:
        self.tagging_nlp = tagging_nlp
        self.ner_nlp = ner_nlp

    @classmethod
    def _lemma(cls, token: SpacyToken):
        if token.lemma_ != "-PRON-":
            return token.lemma_.lower().strip()
        else:
            return token.lower_

    def tag(self, sentence: str) -> List[Token]:
        document = self.tagging_nlp(sentence)   # type: Doc
        return [
            Token(element.orth_, element.tag_, self._lemma(element))
            for element in document   # type: SpacyToken
        ]

    def ner(self, sentence: str) -> Optional[List[Token]]:
        if not self.ner_nlp:
            return None
        document = self.ner_nlp(sentence)  # type: Doc
        return [
            Token(element.orth_, element.tag_, self._lemma(element),
                  '-'.join([element.ent_iob_, element.ent_type_])
                  if element.ent_type_ else None)
            for element in document  # type: SpacyToken
        ]


class Encoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, Token):
            return obj.serialize()
        return super().default(obj)


class App(Bottle):

    def __init__(self, handler: Handler):
        super().__init__()
        self.route('/tag', method='POST', callback=self.make_callback(handler.tag))
        if handler.ner_nlp:
            self.route('/ner', method='POST', callback=self.make_callback(handler.ner))

    @classmethod
    def make_callback(cls, f: Callable[[str], List[Token]]) -> Callable[[], None]:
        def handle():
            sentence = request.json.get('sentence', None)
            if not sentence:
                abort(401, "Missing sentence")
            tokens = f(sentence)
            response.content_type = 'application/json'
            return json.dumps(tokens, cls=Encoder)
        return handle


@click.command()
@click.option('--port', default=9090)
@click.option('--language', default='en')
@click.option('--ner', is_flag=True)
def serve(port: int, language: str, ner: bool) -> None:
    coloredlogs.install(stream=sys.stderr, level=logging.INFO,
                        fmt='%(asctime)s %(name)s %(levelname)s %(message)s')

    logging.info("Loading ...")

    tagging_nlp = spacy.load(language, disable=['ner'])  # type: Language
    tagging_nlp.remove_pipe('parser')
    logging.info("Tagging pipeline: %s", ', '.join(tagging_nlp.pipe_names))

    ner_nlp = None
    if ner:
        ner_nlp = spacy.load(language)  # type: Language
        logging.info("NER pipeline: %s", ', '.join(ner_nlp.pipe_names))

    app = App(Handler(tagging_nlp, ner_nlp))
    logging.info("Serving on port %d ...", port)

    meinheld.set_access_logger(None)
    app.run(host='0.0.0.0', port=port, server="meinheld", workers=16, quiet=True, debug=False)


if __name__ == '__main__':
    serve()
