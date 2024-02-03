#!/usr/bin/env python3

from generator import ChirpShape, lora_symbol, noise_for, LoraConfig
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import tqdm


lora_config = LoraConfig(
	spread_factor=8,
	bandwidth=6000,
	sample_rate=48000,
	shape=ChirpShape.STEPPED,
)

symbol = 20
snr = -60
runs = 15000


signal_time, symbol_shape = lora_symbol(
	lora_config,
	symbol=symbol,
)

_, base_down_chirp_shape = lora_symbol(
	lora_config,
	conj=True,
)

results = np.zeros(runs)
for i in tqdm.tqdm(range(runs), total=runs):

	noise_shape = noise_for(symbol_shape, snr)
	signal_shape = symbol_shape + noise_shape

	dechirped_shape = signal_shape * base_down_chirp_shape

	correlations = np.absolute(np.fft.fft(dechirped_shape)) ** 2

	results[i] = np.argmax(correlations)


symbol_error_rate = np.count_nonzero(results != symbol) / runs


fig, axs = plt.subplots(2)
fig.set_size_inches(18, 11)

fig.suptitle(f"Decoding LORA symbol #{symbol} {runs} times for {lora_config.bit_rate:.0f}bps with SNR={snr}dB @ {lora_config}")

ax = axs[0]
ax.set_title(f"Original signal ({signal_time})")
ax.set(xlabel="Time", ylabel="Amplitude")
ax.plot(signal_time.times, symbol_shape.real, color="blue", alpha=0.5)
ax.plot(signal_time.times, symbol_shape.imag, color="red", alpha=0.2)

ax = axs[1]
ax.set_title(f"Decodings (ser={symbol_error_rate*100:.2f}% @ snr={snr}dB)")
ax.set(xlabel="Symbol", ylabel="Count")
symbols = np.arange(lora_config.symbol_count)
hist, _ = np.histogram(results, bins=np.arange(lora_config.symbol_count + 1))
ax.bar(
	symbols,
	hist,
	color=[
		"green"
		if symbol_iter == symbol else
		"olive"
		for symbol_iter in symbols
	]
)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(runs))

plt.show()
