import math
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

i = 1j


@dataclass
class TimeSlice():
	sample_rate: float
	begin_sample: int
	end_sample: int

	def __post_init__(self):
		if self.begin_sample > self.end_sample or self.sample_rate < 0:
			raise ValueError("Invalid timeslice")

	def __str__(self):
		return f"[{self.begin_sample}, {self.end_sample})"

	@property
	def times(self):
		return self.indices / self.sample_rate

	@property
	def indices(self):
		return np.arange(self.begin_sample, self.end_sample)

	@property
	def sample_count(self):
		return self.end_sample - self.begin_sample

	@property
	def duration(self):
		return self.sample_count / self.sample_rate

	def __and__(self, other: "TimeSlice"):
		if self.end_sample < other.begin_sample or self.begin_sample > other.end_sample:
			raise ValueError("Resulting timeslice would have a gap")
		if self.sample_rate != other.sample_rate:
			raise ValueError("Combining timeslices of different sample-rates currently not implemented")
		return TimeSlice(
			sample_rate=self.sample_rate,
			begin_sample=min(self.begin_sample, other.begin_sample),
			end_sample=max(self.end_sample, other.end_sample),
		)

	def __lshift__(self, other: "TimeSlice"):
		if self.sample_rate != other.sample_rate:
			raise ValueError("Aligning timeslices of different sample-rates currently not implemented")
		return TimeSlice(
			sample_rate=self.sample_rate,
			begin_sample=other.begin_sample - self.sample_count,
			end_sample=other.begin_sample,
		)

	def __rshift__(self, other: "TimeSlice"):
		if self.sample_rate != other.sample_rate:
			raise ValueError("Aligning timeslices of different sample-rates currently not implemented")
		return TimeSlice(
			sample_rate=self.sample_rate,
			begin_sample=other.end_sample,
			end_sample=other.end_sample + self.sample_count,
		)

	def __add__(self, samples: int):
		return TimeSlice(
			sample_rate=self.sample_rate,
			begin_sample=self.begin_sample + samples,
			end_sample=self.end_sample + samples,
		)

	def __sub__(self, samples: int):
		return TimeSlice(
			sample_rate=self.sample_rate,
			begin_sample=self.begin_sample - samples,
			end_sample=self.end_sample - samples,
		)


@dataclass(frozen=True)
class LoraConfig:
	spread_factor: float
	bandwidth: float
	sample_rate: float

	@property
	def symbol_count(self):
		return 2 ** self.spread_factor

	@property
	def symbol_rate(self):
		return self.bandwidth / self.symbol_count

	@property
	def bit_rate(self):
		return self.symbol_rate * self.spread_factor

	@staticmethod
	def create(symbol_count: int, sample_rate: float, symbol_rate: float):
		symbol_samples = sample_rate / symbol_rate
		spread_factor = math.log2(symbol_count)
		bandwidth = symbol_count * sample_rate / symbol_samples
		return LoraConfig(
			spread_factor=spread_factor,
			bandwidth=bandwidth,
			sample_rate=sample_rate,
		)


def lora_symbol(config: LoraConfig, symbol: int = 0, conj: bool = False):
	sf2 = config.symbol_count
	time_slice = TimeSlice(config.sample_rate, 0, math.ceil(sf2 / config.bandwidth * config.sample_rate))
	k = (time_slice.indices + symbol) % sf2 + 1
	magnitude = 1 / math.sqrt(sf2)
	phase = math.tau * k * k / (sf2 * 2)
	if conj:
		phase = -phase
	return time_slice, magnitude * np.exp(i * phase)


def noise(stdev: float, time_slice: TimeSlice):
	return np.random.normal(
		loc=0,
		scale=stdev,
		size=time_slice.sample_count * 2,
	).view(complex)


def noise_for(signal: npt.NDArray, snr: float):
	signal_variance = signal.var()
	noise_variance = signal_variance / level(snr)
	noise_signal = np.random.normal(
		loc=0,
		scale=np.sqrt(noise_variance / 2),
		size=len(signal) * 2,
	).view(complex)
	return noise_signal


def level(db: float):
	return 10 ** (db / 20)


def db(level: float):
	return math.log10(level) / 20
