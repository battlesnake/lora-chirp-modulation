#!/usr/bin/env python3

from generator import lora_symbol, noise_for, LoraConfig
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np


lora_config = LoraConfig(
	spread_factor=7,
	bandwidth=6000,
	sample_rate=48000,
)

symbol = 20
snr = -20


signal_time, symbol_shape = lora_symbol(
	lora_config,
	symbol=symbol,
)

_, base_down_chirp_shape = lora_symbol(
	lora_config,
	conj=True,
)

runs = 100
results = np.zeros(runs)
for i in range(runs):

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
ax.set_title(f"Decodings (ser={symbol_error_rate*100}% @ snr={snr}dB)")
ax.set(xlabel="Symbol", ylabel="Count")
n, bins, patches = ax.hist(results, bins=np.arange(lora_config.symbol_count), color="olive", density=True)
patches[symbol].set_fc("green")
if lora_config.symbol_count > 300:
	ax.plot(0.5 * (bins[1:] + bins[:-1]), n)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))

plt.show()
