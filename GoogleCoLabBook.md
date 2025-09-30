!git clone https://github.com/bassanro/llm-train-education.git
%cd llm-train-education
!git checkout develop
!python -m pip install --upgrade pip
!pip install -r requirements.txt

# optional: install spaCy model
!python -m spacy download en_core_web_sm

# run fine-tune (adjust args as needed)
!python main.py fine-tune
