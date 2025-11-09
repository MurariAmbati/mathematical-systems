"""
Output manager for exporting audio, MIDI, and metadata.

Handles all file export functionality with reproducibility metadata.
"""

from typing import Dict, Any, Optional, List
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import mido
from datetime import datetime


class OutputManager:
    """
    Manages export of audio, MIDI, and metadata files.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize output manager.
        
        Args:
            output_dir: Base directory for outputs (default: current directory)
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_audio(
        self,
        signal: np.ndarray,
        filename: str,
        sample_rate: int = 44100,
        bit_depth: int = 16,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Export audio signal to WAV file.
        
        Args:
            signal: Audio signal (mono or stereo)
            filename: Output filename (with or without .wav extension)
            sample_rate: Sample rate in Hz
            bit_depth: Bit depth (16, 24, or 32)
            metadata: Optional metadata to embed
            
        Returns:
            Path to exported file
        """
        # Ensure filename has .wav extension
        if not filename.endswith('.wav'):
            filename += '.wav'
        
        filepath = self.output_dir / filename
        
        # Determine subtype based on bit depth
        subtype_map = {
            16: 'PCM_16',
            24: 'PCM_24',
            32: 'PCM_32'
        }
        subtype = subtype_map.get(bit_depth, 'PCM_16')
        
        # Ensure signal is in valid range [-1, 1]
        signal = np.clip(signal, -1.0, 1.0)
        
        # Write audio file
        sf.write(
            str(filepath),
            signal,
            sample_rate,
            subtype=subtype
        )
        
        # Export metadata if provided
        if metadata:
            metadata_path = self._export_metadata(
                metadata,
                filepath.stem + '_metadata.json'
            )
            print(f"Metadata exported to: {metadata_path}")
        
        print(f"Audio exported to: {filepath}")
        return filepath
    
    def export_midi(
        self,
        notes: List[Dict[str, Any]],
        filename: str,
        tempo: int = 120,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Export MIDI file.
        
        Args:
            notes: List of note dictionaries with keys:
                   - 'note': MIDI note number (0-127)
                   - 'start': Start time in seconds
                   - 'duration': Duration in seconds
                   - 'velocity': Velocity (0-127, default: 64)
            filename: Output filename (with or without .mid extension)
            tempo: Tempo in BPM
            metadata: Optional metadata to export separately
            
        Returns:
            Path to exported file
        """
        # Ensure filename has .mid extension
        if not filename.endswith('.mid'):
            filename += '.mid'
        
        filepath = self.output_dir / filename
        
        # Create MIDI file
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Set tempo
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))
        
        # Convert notes to MIDI events
        events = []
        for note_info in notes:
            note = int(note_info['note'])
            start = note_info['start']
            duration = note_info['duration']
            velocity = note_info.get('velocity', 64)
            
            # Note on event
            events.append({
                'time': start,
                'type': 'note_on',
                'note': note,
                'velocity': velocity
            })
            
            # Note off event
            events.append({
                'time': start + duration,
                'type': 'note_off',
                'note': note,
                'velocity': 0
            })
        
        # Sort events by time
        events.sort(key=lambda x: x['time'])
        
        # Convert to delta times and add to track
        current_time = 0
        ticks_per_beat = mid.ticks_per_beat
        seconds_per_tick = 60.0 / (tempo * ticks_per_beat)
        
        for event in events:
            # Calculate delta time in ticks
            delta_time = event['time'] - current_time
            delta_ticks = int(delta_time / seconds_per_tick)
            
            # Add MIDI message
            if event['type'] == 'note_on':
                track.append(mido.Message(
                    'note_on',
                    note=event['note'],
                    velocity=event['velocity'],
                    time=delta_ticks
                ))
            else:
                track.append(mido.Message(
                    'note_off',
                    note=event['note'],
                    velocity=event['velocity'],
                    time=delta_ticks
                ))
            
            current_time = event['time']
        
        # Save MIDI file
        mid.save(str(filepath))
        
        # Export metadata if provided
        if metadata:
            metadata_path = self._export_metadata(
                metadata,
                filepath.stem + '_metadata.json'
            )
            print(f"Metadata exported to: {metadata_path}")
        
        print(f"MIDI exported to: {filepath}")
        return filepath
    
    def _export_metadata(
        self,
        metadata: Dict[str, Any],
        filename: str
    ) -> Path:
        """
        Export metadata to JSON file.
        
        Args:
            metadata: Metadata dictionary
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        if not filename.endswith('.json'):
            filename += '.json'
        
        filepath = self.output_dir / filename
        
        # Add timestamp
        metadata['export_timestamp'] = datetime.now().isoformat()
        
        # Write JSON file
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return filepath
    
    def export_metadata(
        self,
        metadata: Dict[str, Any],
        filename: str
    ) -> Path:
        """
        Public method to export metadata.
        
        Args:
            metadata: Metadata dictionary
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        return self._export_metadata(metadata, filename)
    
    def create_reproducibility_metadata(
        self,
        expression: Optional[str] = None,
        generator_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        mappings: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create comprehensive reproducibility metadata.
        
        Args:
            expression: Mathematical expression used
            generator_name: Name of generator used
            parameters: Generator/function parameters
            mappings: Mapping configurations
            seed: Random seed (if applicable)
            **kwargs: Additional metadata fields
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'version': '0.1.0',
            'timestamp': datetime.now().isoformat(),
        }
        
        if expression:
            metadata['expression'] = expression
        
        if generator_name:
            metadata['generator'] = generator_name
        
        if parameters:
            metadata['parameters'] = parameters
        
        if mappings:
            metadata['mappings'] = mappings
        
        if seed is not None:
            metadata['seed'] = seed
        
        # Add any additional fields
        metadata.update(kwargs)
        
        return metadata
    
    def load_metadata(self, filename: str) -> Dict[str, Any]:
        """
        Load metadata from JSON file.
        
        Args:
            filename: Metadata filename
            
        Returns:
            Metadata dictionary
        """
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Metadata file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def export_waveform_data(
        self,
        signal: np.ndarray,
        time: np.ndarray,
        filename: str
    ) -> Path:
        """
        Export raw waveform data as NumPy array.
        
        Args:
            signal: Signal array
            time: Time array
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        if not filename.endswith('.npz'):
            filename += '.npz'
        
        filepath = self.output_dir / filename
        
        np.savez(
            str(filepath),
            signal=signal,
            time=time
        )
        
        print(f"Waveform data exported to: {filepath}")
        return filepath
    
    def load_waveform_data(self, filename: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load waveform data from NumPy file.
        
        Args:
            filename: Waveform data filename
            
        Returns:
            Tuple of (signal, time) arrays
        """
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Waveform data file not found: {filepath}")
        
        data = np.load(str(filepath))
        return data['signal'], data['time']
    
    def set_output_directory(self, directory: str):
        """
        Set the output directory.
        
        Args:
            directory: New output directory path
        """
        self.output_dir = Path(directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory set to: {self.output_dir}")


def create_metadata(
    expression: Optional[str] = None,
    generator: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to create metadata.
    
    Args:
        expression: Mathematical expression
        generator: Generator name
        parameters: Parameters dictionary
        **kwargs: Additional metadata
        
    Returns:
        Metadata dictionary
    """
    manager = OutputManager()
    return manager.create_reproducibility_metadata(
        expression=expression,
        generator_name=generator,
        parameters=parameters,
        **kwargs
    )
