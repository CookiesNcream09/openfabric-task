import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time
import en_core_web_sm
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span


############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    pass

nlp = en_core_web_sm.load()

knowledge_base = {
    "What is science?": "Science is the systematic study of the natural world through observation and experimentation.",
    "Who discovered science?": "Science is not something that was discovered by a single person, but rather a collective effort of many individuals throughout history.",
    "What is physics?": "Physics is the branch of science that deals with the study of matter, energy, and their interactions.",
    "What is biology?": "Biology is the branch of science that deals with the study of living organisms, including their structure, function, growth, evolution, and distribution."
    "How to make a volcano?": "To make a volcano, you can mix baking soda and vinegar to create a chemical reaction that produces carbon dioxide gas, which causes the eruption.",
    "What is the boiling point of water?": "The boiling point of water is 100 degrees Celsius at standard atmospheric pressure.",
    "What is the chemical formula for water?": "The chemical formula for water is H2O.",
    "What is the capital of France?": "I'm sorry, I don't know the answer to that question. Please ask me a question related to science.",

    "What are Newton's laws of motion?": "Newton's laws of motion describe the relationship between an object and the forces acting upon it. The laws are: 1) an object at rest will remain at rest, and an object in motion will remain in motion at a constant velocity, unless acted upon by a net external force; 2) the acceleration of an object is directly proportional to the force applied to it, and inversely proportional to its mass; 3) for every action, there is an equal and opposite reaction.",
    "What is the periodic table?": "The periodic table is a tabular display of the chemical elements, organized on the basis of their atomic structure. The elements are arranged in rows and columns according to their increasing atomic number, which reflects the number of protons in the nucleus of an atom of that element.",
    "What is genetics?": "Genetics is the study of genes, heredity, and genetic variation in living organisms. It involves the study of how traits are passed down from one generation to the next, and how genes and DNA are responsible for the traits that are expressed in an organism.",
    "What is algebra?": "Algebra is a branch of mathematics that deals with mathematical symbols and the rules for manipulating these symbols. It involves solving equations and manipulating variables to find unknown quantities.",
    "What is geology?": "Geology is the study of the Earth's physical structure, properties, and processes, as well as the history of the planet and its life forms. It involves the study of rocks, minerals, fossils, earthquakes, and other geological phenomena.",
    "What is the Solar System?": "The Solar System is the collection of planets, asteroids, comets, and other objects that orbit around the Sun. It includes the eight planets in our own solar system, as well as numerous smaller bodies that are found in the Kuiper Belt and the Oort Cloud.",
    "What is machine learning?": "Machine learning is a field of computer science that involves the use of algorithms and statistical models to enable computer systems to learn from data, without being explicitly programmed. It involves the development of systems that can automatically improve their performance with experience.",
    "What is cosmology?": "Cosmology is the study of the origins, evolution, and structure of the universe as a whole. It involves the study of the large-scale structure of the universe, the cosmic microwave background radiation, dark matter, and dark energy.",
    "What is anatomy?": "Anatomy is the branch of biology that deals with the structure and organization of living organisms. It involves the study of the physical structures of organisms, including their organs, tissues, and cells.",
    "What is a chemical reaction?": "A chemical reaction is a process that leads to the transformation of one set of chemical substances to another. It involves the breaking and forming of chemical bonds between atoms, resulting in the creation of new chemical compounds."

}

# we are creating object phasematcher for matching knowledge base
matcher = PhraseMatcher(nlp.vocab)
patterns = list(nlp.pipe(knowledge_base.keys()))
matcher.add("KnowledgeBase", None, *patterns)


def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        doc = nlp(text)
        # we use the phraseMatcher to match the question to the knowledge base
        matches = matcher(doc)
        if matches:
            # Get the matched question
            matched_question = knowledge_base[nlp.vocab.strings[matches[0][0]]]
            response = matched_question
        else:
            response = "I'm sorry, I don't understand your question."
        output.append(response)

    return SimpleText(dict(text=output))
