from datasets import load_dataset
import torch
import tqdm
from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification

BATCH = 16

def load_imagenet_resnet50(
    path_dataset: str = "imagenet-1k",
    path_model: str = "microsoft/resnet-50",
    sample: int = None,
    sample_seed: int = 42):

    dataset = load_dataset(path_dataset, split='validation', verification_mode='no_checks', trust_remote_code=True)
    if sample is not None:
        dataset = dataset.shuffle(seed=sample_seed)
        dataset = dataset.take(sample)
        
    processor = AutoImageProcessor.from_pretrained(path_model)
    model = AutoModelForImageClassification.from_pretrained(path_model)

    images = [image.convert('RGB') for image in dataset['image']]
    inputs = processor(images, return_tensors='pt')
    inputs['pixel_values'].requires_grad_(True)

    for i in tqdm.trange(len(inputs['pixel_values']) // BATCH):
        outputs = model(pixel_values=inputs['pixel_values'][i*BATCH:(i+1)*BATCH])
        outputs['logits'].max(dim=-1).values.sum().backward()

    i_last = len(inputs['pixel_values']) // BATCH
    outputs = model(pixel_values=inputs['pixel_values'][i_last*BATCH:])
    outputs['logits'].max(dim=-1).values.sum().backward()

    raw_x = inputs['pixel_values'].detach()
    raw_y = inputs['pixel_values'].grad.detach()

    return raw_x, raw_y

def create_perturbations(
    i: int,
    delta: float = None,
    path_dataset: str = "imagenet-1k",
    path_model: str = "microsoft/resnet-50",
    seed: int = 42):

    dataset = load_dataset(path_dataset, split='validation', verification_mode='no_checks', trust_remote_code=True)
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.take(i+1)

    processor = AutoImageProcessor.from_pretrained(path_model)
    model = AutoModelForImageClassification.from_pretrained(path_model)

    image = dataset['image'][i].convert('RGB')
    _input = processor(image, return_tensors='pt')['pixel_values']
    _range = _input.max() - _input.min()
    label = model(pixel_values=_input)['logits'].max(dim=-1).indices.item()

    inputs = [_input.clone().detach()]
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        
        if delta is not None:
            for _ in range(100):
                inputs.append(_input.clone().detach() + delta * _range * torch.randn_like(_input))

        if delta is None:
            for _ in range(1000):
                noise_direction = torch.randn_like(_input.mean(dim=1))[:,None,...]
                noise_direction = noise_direction / torch.sum(noise_direction**2, dim=(2,3), keepdim=True)**.5
                noise_magnitude = 10. * torch.rand(size=(1,)).item() * (_input.shape[2] * _input.shape[3])**.5
                inputs.append(_input.clone().detach() + noise_magnitude * noise_direction)

    inputs = torch.concatenate(inputs, dim=0)
    inputs = torch.nn.Parameter(inputs)
    inputs.requires_grad_(True)

    for i in tqdm.trange(len(inputs) // BATCH):
        outputs = model(pixel_values=inputs[i*BATCH:(i+1)*BATCH])
        outputs['logits'].max(dim=-1).values.sum().backward()
        # outputs['logits'][...,label].sum().backward()

    i_last = len(inputs) // BATCH
    outputs = model(pixel_values=inputs[i_last*BATCH:])
    outputs['logits'].max(dim=-1).values.sum().backward()
    # outputs['logits'][...,label].sum().backward()
        
    raw_x = inputs.detach()
    raw_y = inputs.grad.detach()

    return raw_x, raw_y


