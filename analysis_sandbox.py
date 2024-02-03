#!/usr/bin/env python3

import functools
from generator import lora_symbol, noise_for, LoraConfig
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.scale
import numpy as np
import itertools
import tqdm
import multiprocessing
import math


sample_rate = 96000
bandwidth = sample_rate
runs = 200
symbol = 42

sf_range = np.arange(5, 14 + 1)
snr_range = np.arange(-60, 0 + 1, 1)

results = np.zeros((len(sf_range), len(snr_range), runs))


@functools.cache
def cached_lora_symbol(config: LoraConfig, symbol: int = 0, conj: bool = False):
	return lora_symbol(config, symbol, conj)


def inner_loop(sf_index, sf, others):

	snr_iter, run = others

	snr_index, snr = snr_iter

	lora_config = LoraConfig(
		spread_factor=sf,
		bandwidth=bandwidth,
		sample_rate=sample_rate,
	)

	# symbol = random.randint(0, int(lora_config.symbol_count) - 1)

	_, symbol_shape = cached_lora_symbol(
		lora_config,
		symbol=symbol,
	)

	_, base_down_chirp_shape = cached_lora_symbol(
		lora_config,
		conj=True,
	)

	noise_shape = noise_for(symbol_shape, snr)
	signal_shape = symbol_shape + noise_shape

	dechirped_shape = signal_shape * base_down_chirp_shape

	confidences = np.absolute(np.fft.fft(dechirped_shape)) ** 2

	result = np.argmax(confidences) != symbol

	return sf_index, snr_index, run, result


for sf_index, sf in tqdm.tqdm(list(enumerate(sf_range))):
	for _, snr_index, run, result in multiprocessing.Pool().map(
		functools.partial(inner_loop, sf_index, sf),
		itertools.product(enumerate(snr_range), range(runs)),
		20
	):
		results[sf_index, snr_index, run] = result


symbol_error_rates = np.apply_along_axis(
	lambda errors: np.count_nonzero(errors) / runs,
	2,
	results,
)

print(symbol_error_rates)

bitrates = np.zeros((len(sf_range),))
for sf_index, sf in enumerate(sf_range):
	bitrates[sf_index] = LoraConfig(
		spread_factor=sf,
		bandwidth=sample_rate,
		sample_rate=sample_rate,
	).bit_rate

fig, axs = plt.subplots(2)
fig.set_size_inches(18, 11)

fig.suptitle(f"LORA decodings at various spread-factors and signal-to-noise ratios (runs={runs}, fs={sample_rate}, bw={bandwidth})")

ax = axs[0]
ax.set_title("Symbol error-rate")
ax.set(xlabel="Signal-to-noise (dB)", ylabel="Spread factor")

ax.set_facecolor("black")
cp = ax.contourf(
	snr_range,
	sf_range,
	symbol_error_rates,
	[
		level * math.pow(10, substep / 3)
		for level in [0.001, 0.01, 0.1]
		for substep in range(3)
	] + [1],
	norm=matplotlib.colors.LogNorm(),
	cmap="plasma",
)
plt.colorbar(cp)
plt.grid(True)

ax = axs[1]
ax.set_title("Bitrate vs spread factor")
ax.set(xlabel="Spread factor", ylabel="Bitrate", yscale="log", ylim=(1, np.max(bitrates) * 10))
ax.plot(sf_range, bitrates)
plt.grid(True)

plt.show()
