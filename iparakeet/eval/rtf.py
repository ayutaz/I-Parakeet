"""Real-time factor and padding overhead accounting (paper Sec. 4.1)."""


class RTFMeter:
    def __init__(self) -> None:
        self.audio_seconds = 0.0
        self.processing_seconds = 0.0
        self.padded_seconds = 0.0

    def add(self, audio_seconds: float, processing_seconds: float, padded_seconds: float | None = None) -> None:
        self.audio_seconds += audio_seconds
        self.processing_seconds += processing_seconds
        self.padded_seconds += audio_seconds if padded_seconds is None else padded_seconds

    @property
    def rtf(self) -> float:
        return self.processing_seconds / self.audio_seconds if self.audio_seconds else 0.0

    @property
    def padding_ratio(self) -> float:
        if not self.audio_seconds:
            return 0.0
        return (self.padded_seconds - self.audio_seconds) / self.audio_seconds
