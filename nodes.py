from comfy_api.latest import io
from comfy.sd import CLIP
import nodes
import re

class SDXLCLIPTextEncodeNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SDXLCLIPTextEncodeNode",
            display_name="SDXL CLIPTextEncode",
            category="utils",
            inputs=[
                io.Clip.Input("clip"),
                io.Int.Input("width", default=1024, min=0, max=nodes.MAX_RESOLUTION),
                io.Int.Input("height", default=1024, min=0, max=nodes.MAX_RESOLUTION),
                io.Int.Input("crop_w", default=0, min=0, max=nodes.MAX_RESOLUTION, advanced=True),
                io.Int.Input("crop_h", default=0, min=0, max=nodes.MAX_RESOLUTION, advanced=True),
                io.Int.Input("target_width", default=1024, min=0, max=nodes.MAX_RESOLUTION),
                io.Int.Input("target_height", default=1024, min=0, max=nodes.MAX_RESOLUTION),
                io.String.Input("positive", multiline=True, dynamic_prompts=True),
                io.String.Input("negative", multiline=True, dynamic_prompts=True),
                io.Boolean.Input("use_break", optional=True, default=False, advanced=True),
                ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Conditioning.Output("negative"),
                ]
        )

    @classmethod
    def execute(cls, clip, width, height, crop_w, crop_h, target_width, target_height, positive, negative, use_break) -> io.NodeOutput:
        return io.NodeOutput(
            cls.clip_encode(clip, width, height, crop_w, crop_h, target_width, target_height, positive, use_break),
            cls.clip_encode(clip, width, height, crop_w, crop_h, target_width, target_height, negative, use_break),
        )

    @classmethod
    def clip_encode(cls, clip: CLIP, width: int, height: int, crop_w: int, crop_h: int, target_width: int, target_height: int, text: str, use_break: bool) -> any:
        if use_break:
            break_list = re.split(r"[ ]*BREAK[ ]*,?", text)
            for i, v in enumerate(break_list):
                print(f"Processing chunk {i+1}/{len(break_list)}: '{v.strip().replace('\n', '\\n')}'")
                if i == 0:
                    tokens = clip.tokenize(v.strip())
                else:
                    tokens_add = clip.tokenize(v.strip())
                    tokens["g"] += tokens_add["g"]
                    tokens["l"] += tokens_add["l"]
        else:
            tokens = clip.tokenize(text)
            # tokens["l"] = clip.tokenize(text)["l"]
        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]
        return clip.encode_from_tokens_scheduled(tokens, add_dict={"width": width, "height": height, "crop_w": crop_w, "crop_h": crop_h, "target_width": target_width, "target_height": target_height})

class SDXLClipTextTokenDumpNode(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SDXLClipTextTokenDumpNode",
            display_name="SDXL ClipTextToken Dump",
            category="utils",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("text",multiline=True),
                ],
            outputs=[
                io.String.Output("token_g"),
                io.String.Output("token_l"),
                ]
        )

    @classmethod
    def execute(cls, clip, text) -> io.NodeOutput:
        tokens = clip.tokenize(text)
        all_token_strings_g = []
        all_token_strings_l = []
        try:
            id_to_token_l = {v: k for k, v in clip.tokenizer.clip_l.tokenizer.get_vocab().items()}
            id_to_token_g = {v: k for k, v in clip.tokenizer.clip_g.tokenizer.get_vocab().items()}
            # print("\n--- Tokenized Chunks (G) ---")
            num_chunks_g = len(tokens['g'])
            for i, chunk in enumerate(tokens['g']):
                token_ids_g = [item[0] for item in chunk]
                token_strings_g = [id_to_token_g.get(token_id, "[UNK]") for token_id in token_ids_g]
                # all_token_strings_g.extend(token_strings_g)
                # print(f"Chunk {i+1}/{num_chunks_g}: ", " | ".join(token_strings_g))
                all_token_strings_g.append(f"Chunk {i+1}/{num_chunks_g}: {' '.join(token_strings_g)}")
            # print("\n--- Tokenized Chunks (L) ---")
            num_chunks_l = len(tokens['l'])
            for i, chunk in enumerate(tokens['l']):
                token_ids_l = [item[0] for item in chunk]
                token_strings_l = [id_to_token_l.get(token_id, "[UNK]") for token_id in token_ids_l]
                # all_token_strings_l.extend(token_strings_l)
                # print(f"Chunk {i+1}/{num_chunks_l}: ", " | ".join(token_strings_l))
                all_token_strings_l.append(f"Chunk {i+1}/{num_chunks_l}: {' '.join(token_strings_l)}")
        except Exception as e:
            pass
        return io.NodeOutput("\n".join(all_token_strings_g),"\n".join(all_token_strings_l))
        
