import argparse
import requests
from llama_cpp import Llama
from config import Config
import sys
import os
import torch
import shutil
import re
import speech_recognition as sr
import gc

config = Config('./config/config.json')

TRIGGER_ALIAS=f"automation.{config['automation_name'].replace('-', '_')}"
AUTOMATION_TEMPLATE=f"""
- id: '1681120728300'
  alias: {config['automation_name']}
  description: ''
  trigger: []
  condition:
  - condition: device
    device_id: {config['device_id']}
    domain: media_player
    entity_id: media_player.speaker
    type: is_on
  action:
{{media_templates}}
  mode: single
"""

MEDIA_TEMPLATE="""  - service: media_player.play_media
    target:
      entity_id: media_player.speaker
    data:
      media_content_id: media-source://media_source/media/speak{n}.wav
      media_content_type: audio/x-wav
    metadata:
      title: speak{n}.wav
      thumbnail:
      media_class: music
      children_media_class:
      navigateIds:
      - media_content_type: app
        media_content_id: media-source://media_source
  - wait_for_trigger:
    - platform: state
      entity_id:
      - media_player.speaker
      from: playing
      to: idle
    - platform: state
      entity_id:
      - media_player.speaker
      from: playing
      to: paused"""

class AIModelInstance:
    _instance = None

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = Llama(
                model_path=config['model_path'],
                n_ctx=2048,
                n_threads=int(config['threads']),
                f16_kv=True)
        return cls._instance

    @classmethod
    def reset(cls):
        del cls._instance
        gc.collect()
        cls._instance = None


def request_to_llm(llm: Llama, message: str, temperature: float):
    PROMPT = f'### Human:\n### Instruction:\n\n{message}'
    PROMPT_STOP = ["### Human:"]
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 1900
    completion = llm(
        prompt=PROMPT,
        stop=PROMPT_STOP,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )
    llm_response: str = llm._convert_text_completion_to_chat(completion)['choices'][0]['message']['content']

    for char in ('"#:\''):
        llm_response = llm_response.replace(char, '')
    llm_response = llm_response.replace(u'\u200b', '')
    llm_response = llm_response.replace('\n', ' ')
    return llm_response

def replace_abbreviations(m: re.Match):
    phonetics = ('ae','be','sea','de e','ee','ef','gee','ee ch','ai','je','ke','el','emm','enn','ou','pee','quu','er','es','tee','uu','vee','double uu','ex','epsilon','zed')
    alphabetics = ('a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z')
    to_phonetics = dict(zip(alphabetics, phonetics))
    return ','.join(to_phonetics.get(c.lower(), c) for c in m.group(0))

def _transcript_numbers(number: int) -> str:
    tens = (900, 800, 700, 600, 500, 400, 300, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
    tens_to_phonetics = {
        1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five',
        6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten',
        11: 'Eleven', 12: 'Twelve', 13: 'Thirteen', 14: 'Fourteen',
        15: 'Fifteen', 16: 'Sixteen', 17: 'Seventeen', 18: 'Eighteen',
        19: 'Nineteen', 20: 'Twenty', 30: 'Thirty', 40: 'Forty',
        50: 'Fifty', 60: 'Sixty', 70: 'Seventy', 80: 'Eighty',
        90: 'Ninety', 0: 'Zero', 900: 'Nine hundred', 800: 'Eight hundred',
        700: 'Seven hundred', 600: 'Six hundred', 500: 'Five hundred',
        400: 'Four hundred', 300: 'Three hundred', 200: 'Two hundred',
        100: 'One hundred'}
    ion_phonetics = ('decillion', 'nonillion', 'octillion', 'septillion', 'sextillion', 'quadrillion', 'trillion', 'billion', 'million', 'thousand', '')
    ions = tuple((10**(i*3) or 1 for i in reversed(range(0, 11))))
    ions_to_phonetics = dict(zip(ions, ion_phonetics))

    result = []
    for ion_group in ions:
        hundreds, number = number // ion_group, number % ion_group
        if hundreds == 0:
            continue
        for tenth in tens:
            if hundreds == 0:
                break
            if hundreds // tenth == 1:
                result.append(tens_to_phonetics[tenth])
                hundreds %= tenth
        result.append(ions_to_phonetics[ion_group])
    return ' '.join(result)

def transcript_numbers(m: re.Match) -> str:
    number = int(m.group(0))
    # likely a year
    if len(m.group(0)) == 4:
        return ' '.join((_transcript_numbers(number // 100), _transcript_numbers(number % 100)))
    return _transcript_numbers(number)

def text_to_speech(text: str):
    """ Do text pre-processing for the output sample to sound right-ish """

    text = re.sub(r'\b(?:[a-z]*[A-Z][a-z]*){2,}', replace_abbreviations, text)
    text = re.sub(r'(?:\d+)', transcript_numbers, text)

    if not os.path.isfile(config['local_tts']):
        torch.hub.download_url_to_file(config['audio_model'], config['local_tts'])
    device = torch.device('cpu')
    model = torch.package.PackageImporter(config['local_tts']).load_pickle("tts_models", "model")
    model.to(device)
    for n, i in enumerate(range(0, len(text), 900)):
        file = model.save_wav(text=text[i:i+900], sample_rate=48000, speaker=config['speaker'])
        shutil.move(file, config['media_file'].format(n=n))
    return n

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-text', default='', type=str, help='Pass a question or not if you\'d like to speak up')
    parser.add_argument('-t', '--temperature', default=0.0, type=float, help='How freely the model should interpret your question 0.0 to 1.0 value')
    args = parser.parse_args(sys.argv[1:])

    if args.input_text == '':
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print('Ask your question now:')
            audio_data = r.listen(source)
            print('Recognizing...')
            args.input_text = r.recognize_google(audio_data)
            print(f'You asked for: {args.input_text}')

    if not 0.0 <= args.temperature <= 1.0:
        raise ValueError(f'Temperature should be between 0.0 and 1.0 only, not {args.temperature}')

    return args

def process_query(llm: Llama, input_text: str, temperature: float) -> str:
    llm_response = ''
    # First request sometimes returns a single character, so we need to repeat
    for _ in range(3):
        llm_response = request_to_llm(llm, input_text, temperature)
        if len(llm_response) > 1:
            break
    llm_response = llm_response if len(llm_response) > 1 else 'No response'

    print(input_text)
    print(llm_response)

    samples = text_to_speech(llm_response)
    # Do not send anything to home assistant if it is not configured
    if not config['automation_file']:
        return llm_response

    media_samples = '\n'.join((MEDIA_TEMPLATE.format(n=n) for n in range(samples+1)))
    with open(config['automation_file'], 'w') as query_file:
        query_file.write(AUTOMATION_TEMPLATE.format(media_templates=media_samples))

    # Reload the configuration in home assistant and trigger the google home in home assistant
    headers = {'Authorization': f'Bearer {config["token"]}'}
    requests.post('http://localhost:8123/api/services/automation/reload', headers=headers).raise_for_status()
    requests.post(
        'http://localhost:8123/api/services/automation/trigger',
        headers=headers, json={"entity_id": TRIGGER_ALIAS}).raise_for_status()

    return llm_response

def main():
    """ Perform one-shot query to model """
    args = parse_args()
    process_query(AIModelInstance.get_instance(), args.input_text, args.temperature)


if __name__ == '__main__':
    main()

