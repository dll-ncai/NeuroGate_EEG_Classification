from datasets import pipeline as pl
from .channels import CHANNELS, PAIRS, NMT_CHANNELS, NMT_PAIRS

def get_scratch_pl(dataset = 'TUH', mins = 1):
    pipeline = pl.Pipeline()
    if (dataset == 'TUH'):
        pipeline.add(pl.CropData(0, mins * 60))
        pipeline.add(pl.ReduceChannels())
        pipeline.add(pl.BipolarRef())
    elif (dataset == 'NMT'):
        pipeline.add(pl.CropData(60, (1+mins) * 60))
        pipeline.add(pl.ReduceChannels(channels= NMT_CHANNELS))
        pipeline.add(pl.BipolarRef(pairs=NMT_PAIRS, channels= NMT_CHANNELS))
    pipeline.add(pl.ResampleData(100))
    pipeline.add(pl.ClipAbsData(100))
    pipeline.add(pl.HighPassFilter(1.0))
    pipeline.add(pl.NotchFilter(60))
    pipeline.add(pl.Scale(1e6))
    return pipeline

def get_scratch_updated_pl(dataset = 'TUH', mins = 1):
    pipeline = pl.Pipeline()
    if (dataset == 'TUH'):
        pipeline.add(pl.CropData(0, mins * 60))
        pipeline.add(pl.ReduceChannels())
        pipeline.add(pl.BipolarRef())
    elif (dataset == 'NMT'):
        pipeline.add(pl.CropData(60, (1+mins) * 60))
        pipeline.add(pl.HighPassFilter(1.0,45))
        pipeline.add(pl.ReduceChannels(channels= NMT_CHANNELS))
        # pipeline.add(pl.ArtifactRemoval(0.5))
        pipeline.add(pl.BipolarRef(pairs=NMT_PAIRS, channels= NMT_CHANNELS))
    pipeline.add(pl.ResampleData(100))
    pipeline.add(pl.ClipAbsData(100))
    pipeline.add(pl.NotchFilter(60))
    pipeline.add(pl.Scale(1e2))
    return pipeline

def get_wavenet_pipeline(dataset = 'TUH'):
    ''' Returns the preprocessing pipeline for the Wavenet model
    '''
    pipeline = pl.Pipeline()
    if (dataset == 'TUH'):
        pipeline.add(pl.CropData(0, 60))
        pipeline.add(pl.ReduceChannels())
        pipeline.add(pl.BipolarRef())
    elif (dataset == 'NMT'):
        pipeline.add(pl.CropData(60, 120))
        pipeline.add(pl.ReduceChannels(channels= NMT_CHANNELS))
        pipeline.add(pl.BipolarRef(pairs=NMT_PAIRS, channels= NMT_CHANNELS))
    pipeline.add(pl.ClipAbsData(100))
    pipeline.add(pl.ResampleData(250))
    pipeline.add(pl.HighPassFilter(1.0))
    pipeline.add(pl.NotchFilter(60))
    pipeline.add(pl.Scale(1e6))
    return pipeline
