import open_clip, re

model, _, _ = open_clip.create_model_and_transforms(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)

names = [n for n, _ in model.named_parameters()]

print("--- FIRST 20 ---")
for n in names[:20]:
    print(n)

print("\n--- LAST 20 ---")
for n in names[-20:]:
    print(n)

print("\n--- TOTAL PARAMS ---", sum(p.numel() for p in model.parameters()))

blocks = set(re.findall(r'visual\.transformer\.resblocks\.(\d+)', ' '.join(names)))
print("--- VISUAL RESBLOCKS ---", "count:", len(blocks), "| indices:", sorted(int(x) for x in blocks))
