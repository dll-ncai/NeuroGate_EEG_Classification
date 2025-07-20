''' Defines all the preprocessing functions and applies them
    to the data in the pipeline
'''


import mne
import numpy as np
from .channels import CHANNELS, PAIRS, NMT_CHANNELS, NMT_PAIRS

class Preprocess:
    def func(self, data):
        # Do something to the data
        # This is to be overloaded always
        return data
    def apply(self, data):
        ''' Applies the preprocessing pipeline to the data
            INPUT:
                data - EEG - data to be preprocessed
            OUTPUT:
                data - EEG - preprocessed data
        '''
        return self.func(data)
    def get_id(self):
        ''' Returns the ID of the preprocessing function
        '''
        return self.__class__.__name__

class ReduceChannels(Preprocess):
    ''' Reducing the number of channels to the 21 channels in use
        Takes in raw data in mne format
        Returns raw data in mne format with 21 channels only
    '''
    def __init__(self, channels=CHANNELS):
        self.channels = channels
    def func(self, data):
        return data.pick(self.channels)

class ClipData(Preprocess):
    ''' Responsible for Clipping the data inside a fixed voltage range
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data clipped between 0 and absclipx10^-6
    '''
    def __init__(self, absclip):
        self.absclip = absclip
    def func(self, data):
        return data.apply_function(lambda data: np.clip(data, 0, 0.000001*self.absclip))
    def get_id(self):
        return f'{self.__class__.__name__}_{self.absclip}'


class ClipAbsData(Preprocess):
    ''' Responsible for Clipping the data inside a fixed voltage range
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data clipped between -absclipx10^-6 and absclipx10^-6
    '''
    def __init__(self, absclip):
        self.absclip = absclip
    def func(self, data):
        return data.apply_function(lambda data: np.clip(data, -0.000001*self.absclip, 0.000001*self.absclip))
    def get_id(self):
        return f'{self.__class__.__name__}_{self.absclip}'

class ResampleData(Preprocess):
    ''' Responsible for resampling the data to 100 Hz
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data resampled to 100 Hz
    '''
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
    def func(self, data):
        sfreq = data.info['sfreq']
        if (sfreq == self.sample_rate):
            return data
        return data.resample(self.sample_rate)
    def get_id(self):
        return f'{self.__class__.__name__}_{self.sample_rate}'

class CropData(Preprocess):
    ''' Responsible for cropping the data to the specified time range
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data cropped to the specified time range
    '''
    def __init__(self, tmin, tmax):
        self.tmin = tmin
        self.tmax = tmax
        self.time_span = tmax - tmin
    def func(self, data):
        return data.crop(tmin=self.tmin, tmax=self.tmax, include_tmax=False)
    def get_id(self):
        return f'{self.__class__.__name__}_{self.time_span}_{self.tmin}'

class PaddedCropData(CropData):
    ''' Responsible for cropping the data to the specified time range.
        If duration < tmax, flips the data, and appends the flipped data to
        the end.
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data cropped to the specified time range
    '''
    def __init__(self, tmin, tmax, reverse=False):
        self.tmin = tmin
        self.tmax = tmax
        self.time_span = tmax - tmin
        self.reverse = reverse
    def func(self, data):
        data.crop(tmin=self.tmin)
        if data.n_times / data.info["sfreq"] >= self.tmax:
            return data.crop(tmin=0, tmax=self.tmax - self.tmin, include_tmax=False)
        else:
            while data.n_times / data.info["sfreq"] < self.tmax:
                data_only, _ = data[:]
                reversed = np.flip(data_only, axis = 1)
                info = data.info
                if self.reverse:
                    data_only = np.concatenate([data_only, reversed], axis=1)
                else:
                    data_only = np.concatenate([data_only, data_only], axis=1)
                data = mne.io.RawArray(data_only, info)
            return data.crop(tmin=0, tmax=self.tmax - self.tmin, include_tmax=False)
    def get_id(self):
        return f'{self.__class__.__name__}_{self.time_span}'

class FilterOut(Preprocess):
    '''Reponsible for not processing files below a certain length
    '''
    def __init__(self, min_len = 6, max_len=50):
        self.min_len = min_len
        self.max_len = max_len
    def func(self, data):
        if data.n_times / data.info['sfreq'] > self.max_len * 60:
            raise ValueError(f"Data length is longer than the maximum allowed length of {self.max_len} minutes.")
        if data.n_times / data.info['sfreq'] < self.min_len * 60:
            raise ValueError(f"Data length is shorter than the minimum allowed length of {self.min_len} minutes.")
        return data
    def get_id(self):
        return f'{self.__class__.__name__}_{self.min_len}_{self.max_len}'

class BandPassFilter(Preprocess):
    ''' Responsible for applying a band-pass filter to the data
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data with a band-pass filter applied
    '''
    def __init__(self, l_freq, h_freq, method='iir', iir_params=None, fir_design='firwin'):
        '''
        Args:
            l_freq (float): Lower cutoff frequency in Hz
            h_freq (float): Upper cutoff frequency in Hz
            method (str): Filtering method, e.g. 'iir' or 'fir'. Default: 'iir'.
            iir_params (dict, optional): Parameters for IIR filter (order, ftype, output). Default: Butterworth order 4 SOS.
            fir_design (str): FIR filter design to use when method='fir'. Default: 'firwin'.
        '''
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.method = method
        # Default IIR params: 4th order butterworth SOS
        self.iir_params = iir_params or dict(order=4, ftype='butter', output='sos')
        self.fir_design = fir_design

    def func(self, data):
        sfreq = data.info['sfreq']
        nyquist = sfreq / 2.0
        # Ensure cutoff frequencies are valid
        if not (0 < self.l_freq < self.h_freq < nyquist):
            raise ValueError(f"Cutoff frequencies must satisfy 0 < l_freq < h_freq < Nyquist ({nyquist} Hz)")

        return data.filter(
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            method=self.method,
            iir_params=self.iir_params if self.method == 'iir' else None,
            fir_design=self.fir_design if self.method == 'fir' else None,
            verbose='error'
        )

    def get_id(self):
        return f"{self.__class__.__name__}_{self.l_freq}_{self.h_freq}_{self.method}"

class ChebyshevFilter(Preprocess):
    ''' Responsible for applying a Chebyshev Type I band-pass filter to the data
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data with a Chebyshev band-pass filter applied
    '''
    def __init__(self, l_freq, h_freq, order, ripple=1.0):
        '''
        Args:
            l_freq (float): Lower cutoff frequency in Hz
            h_freq (float): Upper cutoff frequency in Hz
            order (int): Filter order
            ripple (float): Passband ripple in dB (default: 1.0 dB)
        '''
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.order = order
        self.ripple = ripple
        self.iir_params = dict(order=self.order, ftype='cheby1', rp=self.ripple, output='sos')

    def func(self, data):
        sfreq = data.info['sfreq']
        nyquist = sfreq / 2.0
        # Validate cutoff frequencies
        if not (0 < self.l_freq < self.h_freq < nyquist):
            raise ValueError(f"Cutoff frequencies must satisfy 0 < l_freq < h_freq < Nyquist ({nyquist} Hz)")

        return data.filter(
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            method='iir',
            iir_params=self.iir_params,
            verbose='error'
        )

    def get_id(self):
        return f"{self.__class__.__name__}_{self.l_freq}_{self.h_freq}_ord{self.order}_r{self.ripple}"

class NotchFilter(Preprocess):
    ''' Responsible for applying a notch filter to the data
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data with a notch filter applied
    '''
    def __init__(self, freqs, fir_design='firwin'):
        self.freqs = freqs
        self.fir_design = fir_design
    def func(self, data):
        # Check the sampling rate and calculate the Nyquist frequency
        sfreq = data.info['sfreq']
        nyquist_freq = sfreq / 2
        if (self.freqs < nyquist_freq):
            return data.notch_filter(self.freqs, fir_design=self.fir_design, verbose='error')
        return data
    def get_id(self):
        return f'{self.__class__.__name__}_{self.freqs}'

class ArtifactRemoval(Preprocess):
    ''' Responsible for applying artifact removal to the data
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data with a ICA applied
    '''
    def __init__(self, threshold):
        self.threshold = threshold
    def func(self, data):
        montage =  mne.channels.make_standard_montage('standard_1020')
        data.set_montage(montage, match_case=False,verbose=False)

        ica = mne.preprocessing.ICA(method="picard", max_iter="auto", random_state=56,verbose=False)
        ica.fit(data,verbose=False)

        muscle_idx_auto, scores = ica.find_bads_muscle(data,verbose=False)
        badIndexes = np.where(np.array(scores) > np.median(scores)*self.threshold)[0].tolist()

        ica.exclude = badIndexes
    # print(f"Automatically found muscle artifact ICA components: {badIndexes}")
        ica.apply(data,verbose=False)
        return data
    def get_id(self):
        return f'{self.__class__.__name__}_{self.threshold}'

class Scale(Preprocess):
    ''' Responsible for scaling the data by a fixed numer
        Inputs: Raw EEG in MNE format
        Outputs: Raw EED Data that is scaled
    '''
    def __init__(self, scale):
        self.scale = scale

    def func(self, data):
        data._data *= self.scale
        return data

    def get_id(self):
        return f'{self.__class__.__name__}_{self.scale}'

class BipolarRef(Preprocess):
    ''' Responsible for applying a bipolar reference to the data
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data with a bipolar reference applied
    '''
    def __init__(self, pairs=PAIRS, channels=CHANNELS):
        self.pairs = pairs
        self.channels=channels
    def func(self, data):
        for anode, cathode in self.pairs:
            data = mne.set_bipolar_reference(data.load_data(), anode=[anode], cathode=[cathode], ch_name=f'{anode}-{cathode}', drop_refs=False, copy=True, verbose=False)
        data.drop_channels(ch_names=self.channels)
        return data

class MinMax(Preprocess):
    '''Reponsible for performing channel specific
    min-max normalization.
    '''
    def func(self, data):
        data_only, _ = data[:]

        min_vals = np.min(data_only, axis=1, keepdims=True)
        max_vals = np.max(data_only, axis=1, keepdims=True)

        normed = (data_only - min_vals) / (max_vals - min_vals + np.finfo(float).eps)
        info = data.info
        out = mne.io.RawArray(normed,  info, verbose='error')
        return out

class Reverse(Preprocess):
    ''' Responsible for reversing the data
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data reversed
    '''
    def func(self, data):
        # Get the data as a NumPy array
        data_only, _ = data[:]

        # Reverse the time sequence of the data
        data_reversed = np.flip(data_only, axis=1)

        # Create a new Raw object with the reversed data
        info = data.info
        raw_reversed = mne.io.RawArray(data_reversed, info, verbose='error')
        return raw_reversed

class ZScoreNormalization(Preprocess):
    ''' Performs Z-score normalization using mean and std computed on the first batch
        Ensures consistent scaling across the entire recording by storing parameters
    '''
    def __init__(self, mean, std):
        # Will be set on first call to func()
        self.mean = mean
        self.std = std

    def func(self, data):
        # Extract data array (channels x times)
        data_array, _ = data[:]
        # Apply normalization
        normalized = (data_array - self.mean) / (self.std + np.finfo(float).eps)
        # Return new RawArray with same info
        return mne.io.RawArray(normalized, data.info, verbose='error')

    def get_id(self):
        return f"{self.__class__.__name__}_{self.mean}_{self.std}"

class Pipeline(Preprocess):
    ''' Pipeline class defines the preprocessing pipeline for the EEG data.
        Keeps the pipeline for preprocessing the data
    '''
    def __init__(self):
        ''' Constructor Function
            INPUT:
                pipeline - list - list of functions to be applied to the data
        '''

        self.pipeline = []
        self.sampling_rate = -1
        self.time_span = -1
        self.channels = -1

    def __iter__(self):
        ''' Returns the iterator for the pipeline
        '''
        return iter(self.pipeline)

    def add(self, func):
        ''' Adds a function to the pipeline
            INPUT:
                func - function - function to be added to the pipeline
        '''
        if (func.__class__.__name__ == 'ResampleData'):
            self.sampling_rate = func.sample_rate
        if (func.__class__.__name__ in ['CropData', 'PaddedCropData']):
            self.time_span = func.time_span
        if (func.__class__.__name__ == 'ReduceChannels'):
            self.channels = len(func.channels)
        if (func.__class__.__name__ == 'BipolarRef'):
            self.channels = len(func.pairs)
        self.pipeline.append(func)

    def __add__(self, pipeline):
        ''' Adds a function to the pipeline
            INPUT:
                pipeline - function - function to be added to the pipeline
                pipeline - another list to be added to the pipeline
        '''
        new_pipeline = Pipeline()
        new_pipeline.pipeline = self.pipeline + pipeline.pipeline
        if (pipeline.sampling_rate != -1):
            new_pipeline.sampling_rate = pipeline.sampling_rate
        if (pipeline.time_span != -1):
            new_pipeline.time_span = pipeline.time_span
        if (pipeline.channels != -1):
            new_pipeline.channels = pipeline.channels
        return new_pipeline

    def func(self, data):
        ''' Applies the pipeline to the data
            INPUT:
                data - EEG - data to be preprocessed
            OUTPUT:
                data - EEG - preprocessed data
        '''
        for func in self.pipeline:
            data = func.func(data)
        return data

    def get_id(self):
        return super().get_id() + '_' + '_'.join([func.get_id() for func in self.pipeline])

class MultiPipeline():
    '''MultiPipeline class defines the preprocessing pipeline for the EEG data.
       Combines multiple pipelines to make 1 pipeline
    '''
    def __init__(self, pipelines = []):
        ''' Constructor Function
            INPUT:
                pipelines - list - list of pipelines to be combined
        '''
        self.pipeline = []
        self.sampling_rate = -1
        self.time_span = -1
        self.channels = -1
        if len(pipelines) > 0:
            sample_rate = pipelines[0].sampling_rate
            time_span = pipelines[0].time_span
            channels = pipelines[0].channels
            for pipeline in pipelines:
                if (pipeline.sampling_rate != sample_rate):
                    raise ValueError("Sampling rates do not match")
                if (pipeline.time_span != time_span):
                    raise ValueError("Time spans do not match")
                if (pipeline.channels != channels):
                    raise ValueError("Number of channels do not match")

                self.pipeline.append(pipeline)
            self.sampling_rate = sample_rate
            self.time_span = time_span
            self.channels = channels
        self.len = len(pipelines)

    def __len__(self):
        ''' Returns the length of the pipeline
        '''
        return self.len

    def __iter__(self):
        ''' Returns the iterator for the pipeline
        '''
        return iter(self.pipeline)


    def add(self, pipeline):
        ''' Adds a pipeline to the MultiPipeline
            INPUT:
                pipeline - Pipeline - pipeline to be added to the MultiPipeline
        '''
        if (self.sampling_rate != -1 and self.sampling_rate != pipeline.sampling_rate):
            raise ValueError("Sampling rates do not match")
        if (self.time_span != -1 and self.time_span != pipeline.time_span):
            raise ValueError("Time spans do not match")
        self.pipeline.append(pipeline)
        self.len = len(self.pipeline)

    def __add__(self, pipeline):
        ''' Adds a pipeline to the MultiPipeline
            INPUT:
                pipeline - Pipeline - pipeline to be added to the MultiPipeline
        '''
        newpipeline = MultiPipeline(self.pipeline)
        if pipeline.__class__.__name__ == 'Pipeline':
            newpipeline.add(pipeline)
        elif pipeline.__class__.__name__ == 'MultiPipeline':
            for pipe in pipeline.pipeline:
                newpipeline.add(pipe)
        else:
            raise ValueError("Invalid pipeline")
        return newpipeline

    def get_id(self):
        return 'MULTI_' + '_'.join([pipeline.get_id() for pipeline in self.pipeline])

def neurogate_pipeline(dataset='TUH', length_minutes=10, max_len=25):
    '''Returns a general pipeline that retains most of the recording length
    '''
    pipeline = Pipeline()
    pipeline.add(FilterOut(min_len=6, max_len=max_len))
    if (dataset == 'TUH'):
        pipeline.add(ReduceChannels())
        pipeline.add(BipolarRef())
    elif (dataset == 'NMT'):
        pipeline.add(ReduceChannels(channels= NMT_CHANNELS))
        pipeline.add(BipolarRef(pairs=NMT_PAIRS, channels= NMT_CHANNELS))
    if dataset == 'TUH':
        pipeline.add(NotchFilter(60))
    elif dataset == 'NMT':
        pipeline.add(NotchFilter(50))
    pipeline.add(ResampleData(50))
    pipeline.add(ClipAbsData(100))
    pipeline.add(PaddedCropData(60, 60 + length_minutes * 60, reverse=True))
    pipeline.add(Scale(1e6))
    return pipeline
