# LORA

Experiment sandbox for playing with CHIRP modulation and developing some intuition for it.

## Shannon-Hartley theorem

    C = B log₂( 1 + S/N )

    C : Channel capacity (theoretical maximum information rate, bit/s) [sometimes as I]
    B : Bandwidth (passband width, Hz)
    S : Mean power over the bandwidth (W / V²) [sometimes as C]
    N : Average noise power (same units as S)
    S/N : Signal-to-noise ratio

## Frequency-shift CHIRP modulation: The LoRa Modulation (IEEE doc #8067462)

    Duration of a chip = T = 1 / B

    Duration of a chirp = Tₛ = 2^SF T = 2^SF / B

    Frequency shift per chip = B / 2^SF

i.e. frequency = [ ⌊t / T⌋ (B / 2^SF) ] mod B
