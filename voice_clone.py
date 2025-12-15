import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from utils import get_device
import langid
import argparse

multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=get_device())

def clone_voice(text, reference_audio_path, output_path):
    language_id = langid.classify(text)[0]
    print(f"Detected language: {language_id}")
    wav = multilingual_model.generate(text, audio_prompt_path=reference_audio_path, language_id=language_id)
    ta.save(output_path, wav, multilingual_model.sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clone voice using Chatterbox Multilingual TTS.")
    parser.add_argument("--text", type=str, required=True, help="Text to be synthesized")
    parser.add_argument("--reference_audio_path", type=str, required=True, help="Path to the reference audio file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the cloned voice audio file")
    args = parser.parse_args()

    clone_voice(
        text=args.text,
        reference_audio_path=args.reference_audio_path,
        output_path=args.output_path
    )
