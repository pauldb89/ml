from argparse import ArgumentParser

import torch
from transformers import AutoTokenizer
from transformers import CLIPTextModel
from transformers import CLIPTokenizerFast
from transformers import RobertaTokenizer
from transformers import RobertaTokenizerFast
from transformers import T5EncoderModel
from transformers import T5TokenizerFast


def main():
	parser = ArgumentParser()
	parser.add_argument(
		"--model",
		type=str,
		choices=[
			"openai/clip-vit-base-patch32",
			"google/t5-v1_1-small",
			"google/t5-v1_1-base",
			"google/t5-v1_1-large",
			"google/t5-v1_1-xl",
			"google/t5-v1_1-xxl",
		],
		help="Model to use",
	)
	args = parser.parse_args()

	# tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
	# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
	# inputs = tokenizer(["a photo of cats", "a photo of a dog"], padding=True, return_tensors="pt")
	# print("inputs", inputs)
	# outputs = model.get_text_features(**inputs)
	# print("outputs", outputs)
	# print("memory allocated", torch.cuda.memory_allocated() / 10**9)

	tokenizer_class = T5TokenizerFast if args.model.startswith("google") else CLIPTokenizerFast
	model_class = T5EncoderModel if args.model.startswith("google") else CLIPTextModel

	tokenizer = tokenizer_class.from_pretrained(args.model)
	model = model_class.from_pretrained(args.model).cuda()
	model.eval()

	words = ["athazagoraphobia", "a", "photo", "of", "cats", "dog", "a photo of cats", "a photo of a dog"]

	with torch.inference_mode():
		inputs = tokenizer(words, padding=True, return_tensors="pt")
		print("inputs tokenized")
		inputs = {k: v.cuda() for k, v in inputs.items()}
		print(inputs["attention_mask"])
	
		outputs = model(**inputs)
		print("outputs", outputs.last_hidden_state.size())
		print("memory allocated", torch.cuda.memory_allocated() / 10 ** 9)


if __name__ == "__main__":
	main()
