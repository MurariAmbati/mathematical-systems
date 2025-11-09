"""
Command-line interface for the Math-Music-Engine.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import numpy as np

from ..parser import ExpressionParser, expression_to_function
from ..core import (
    FunctionEngine,
    MappingEngine,
    MappingMode,
    Scale,
    Oscillator,
    WaveformType,
    ADSR,
    OutputManager
)
from ..generators import (
    FourierSynth,
    FibonacciRhythm,
    ChaosModulator,
    FractalMelody,
    PrimeSequence,
    RandomWalk
)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description='Math-Music-Engine: Generate music from mathematical functions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from mathematical expression
  mathmusic generate --function "sin(2*pi*440*t)" --duration 5 --output sine.wav
  
  # Use a generator
  mathmusic generate --generator fourier --harmonics 1,0.5,0.25 --duration 10
  
  # Fibonacci rhythm
  mathmusic generate --generator fibonacci_rhythm --tempo 120 --duration 8
  
  # Chaos modulator
  mathmusic generate --generator lorenz --duration 15 --output chaos.wav
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate audio from functions')
    
    # Input source (mutually exclusive)
    input_group = generate_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--function', '-f',
        type=str,
        help='Mathematical function expression (e.g., "sin(2*pi*440*t)")'
    )
    input_group.add_argument(
        '--generator', '-g',
        type=str,
        choices=['fourier', 'fibonacci_rhythm', 'lorenz', 'rossler', 'logistic',
                'fractal', 'prime', 'random_walk'],
        help='Use a predefined generator'
    )
    
    # Common parameters
    generate_parser.add_argument(
        '--duration', '-d',
        type=float,
        default=10.0,
        help='Duration in seconds (default: 10.0)'
    )
    generate_parser.add_argument(
        '--output', '-o',
        type=str,
        default='output.wav',
        help='Output filename (default: output.wav)'
    )
    generate_parser.add_argument(
        '--sample-rate', '-sr',
        type=int,
        default=44100,
        help='Sample rate in Hz (default: 44100)'
    )
    
    # Mapping parameters
    generate_parser.add_argument(
        '--freq-min',
        type=float,
        default=220.0,
        help='Minimum frequency in Hz (default: 220.0)'
    )
    generate_parser.add_argument(
        '--freq-max',
        type=float,
        default=880.0,
        help='Maximum frequency in Hz (default: 880.0)'
    )
    generate_parser.add_argument(
        '--scale',
        type=str,
        choices=['chromatic', 'major', 'minor', 'pentatonic_major', 'pentatonic_minor'],
        help='Quantize to musical scale'
    )
    generate_parser.add_argument(
        '--root-note',
        type=str,
        default='A4',
        help='Root note for scale (default: A4)'
    )
    
    # Generator-specific parameters
    generate_parser.add_argument(
        '--harmonics',
        type=str,
        help='Comma-separated harmonic amplitudes for Fourier synth (e.g., "1,0.5,0.25")'
    )
    generate_parser.add_argument(
        '--tempo',
        type=float,
        default=120.0,
        help='Tempo in BPM for rhythm generators (default: 120.0)'
    )
    generate_parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    
    # Visualization
    generate_parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    generate_parser.add_argument(
        '--export-metadata',
        action='store_true',
        help='Export metadata JSON file'
    )
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available generators and scales')
    list_parser.add_argument(
        'type',
        nargs='?',
        choices=['generators', 'scales', 'all'],
        default='all',
        help='What to list (default: all)'
    )
    
    return parser


def generate_from_function(args: argparse.Namespace) -> None:
    """Generate audio from mathematical function."""
    print(f"Generating audio from function: {args.function}")
    
    # Parse and create function
    func = expression_to_function(args.function)
    
    # Create engines
    func_engine = FunctionEngine(sample_rate=args.sample_rate)
    mapper = MappingEngine(
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        sample_rate=args.sample_rate
    )
    
    # Sample function
    print(f"Sampling function for {args.duration} seconds...")
    values = func_engine.uniform_sample(func, args.duration, normalize=True)
    
    # Map to frequencies
    scale_enum = Scale[args.scale.upper()] if args.scale else None
    frequencies = mapper.map_to_frequency(
        values,
        mode=MappingMode.LOGARITHMIC,
        scale=scale_enum,
        root_note=args.root_note
    )
    
    # Generate audio using oscillator
    print("Synthesizing audio...")
    osc = Oscillator(sample_rate=args.sample_rate)
    signal = osc.generate(
        WaveformType.SINE,
        frequencies,
        duration=args.duration
    )
    
    # Export
    output_mgr = OutputManager()
    output_mgr.export_audio(signal, args.output, sample_rate=args.sample_rate)
    
    if args.export_metadata:
        metadata = output_mgr.create_reproducibility_metadata(
            expression=args.function,
            parameters={
                'duration': args.duration,
                'sample_rate': args.sample_rate,
                'freq_min': args.freq_min,
                'freq_max': args.freq_max
            },
            mappings={
                'mode': 'logarithmic',
                'scale': args.scale,
                'root_note': args.root_note
            }
        )
        output_mgr.export_metadata(metadata, Path(args.output).stem + '_metadata.json')
    
    if args.visualize:
        from ..visualization import plot_waveform, plot_frequency_spectrum
        print("Creating visualizations...")
        plot_waveform(signal, args.sample_rate, duration=min(1.0, args.duration),
                     output_path=Path(args.output).stem + '_waveform.png')
        plot_frequency_spectrum(signal, args.sample_rate,
                               output_path=Path(args.output).stem + '_spectrum.png')
    
    print(f"✓ Audio generated successfully: {args.output}")


def generate_from_generator(args: argparse.Namespace) -> None:
    """Generate audio using a predefined generator."""
    print(f"Using generator: {args.generator}")
    
    output_mgr = OutputManager()
    
    if args.generator == 'fourier':
        # Fourier synthesis
        if args.harmonics:
            harmonics = [float(x) for x in args.harmonics.split(',')]
        else:
            harmonics = [1.0, 0.5, 0.25, 0.125]  # Default
        
        gen = FourierSynth(harmonics=harmonics, sample_rate=args.sample_rate)
        signal = gen.generate(
            fundamental_freq=args.freq_min,
            duration=args.duration
        )
        metadata = gen.get_metadata()
        
    elif args.generator == 'fibonacci_rhythm':
        # Fibonacci rhythm
        gen = FibonacciRhythm(sample_rate=args.sample_rate)
        pattern = gen.generate_rhythm_pattern(num_beats=8, subdivision=16)
        signal = gen.generate_pulse_train(pattern, args.tempo, args.duration)
        metadata = gen.get_metadata()
        
    elif args.generator in ['lorenz', 'rossler', 'logistic']:
        # Chaos modulator
        gen = ChaosModulator(sample_rate=args.sample_rate)
        signal = gen.generate(args.generator, args.duration)
        metadata = gen.get_metadata(args.generator, {})
        
    elif args.generator == 'fractal':
        # Fractal melody
        gen = FractalMelody(sample_rate=args.sample_rate)
        intervals = gen.generate_koch_curve_melody(iterations=3)
        notes = gen.intervals_to_notes(intervals, base_note=60)
        
        # Convert to audio (simple implementation)
        osc = Oscillator(sample_rate=args.sample_rate)
        note_duration = args.duration / len(notes)
        signal = np.zeros(int(args.duration * args.sample_rate))
        
        for i, note in enumerate(notes):
            start = int(i * note_duration * args.sample_rate)
            end = int((i + 1) * note_duration * args.sample_rate)
            freq = 440 * (2 ** ((note - 69) / 12))
            note_signal = osc.generate(WaveformType.SINE, freq, duration=note_duration)
            signal[start:end] = note_signal[:end-start]
        
        metadata = gen.get_metadata('koch_curve', {'iterations': 3})
        
    elif args.generator == 'prime':
        # Prime sequence
        gen = PrimeSequence(sample_rate=args.sample_rate)
        melody = gen.prime_melody(num_notes=50, method='gaps')
        
        # Convert to audio
        osc = Oscillator(sample_rate=args.sample_rate)
        note_duration = args.duration / len(melody)
        signal = np.zeros(int(args.duration * args.sample_rate))
        
        for i, note in enumerate(melody):
            start = int(i * note_duration * args.sample_rate)
            end = int((i + 1) * note_duration * args.sample_rate)
            freq = 440 * (2 ** ((note - 69) / 12))
            note_signal = osc.generate(WaveformType.SINE, freq, duration=note_duration)
            signal[start:end] = note_signal[:end-start]
        
        metadata = gen.get_metadata('prime_melody', {'num_notes': 50})
        
    elif args.generator == 'random_walk':
        # Random walk
        gen = RandomWalk(sample_rate=args.sample_rate, seed=args.seed)
        signal = gen.brownian_motion(args.duration)
        metadata = gen.get_metadata('brownian_motion', {})
        
    else:
        print(f"Unknown generator: {args.generator}")
        return
    
    # Export audio
    output_mgr.export_audio(signal, args.output, sample_rate=args.sample_rate)
    
    if args.export_metadata:
        output_mgr.export_metadata(metadata, Path(args.output).stem + '_metadata.json')
    
    if args.visualize:
        from ..visualization import plot_waveform, plot_frequency_spectrum
        print("Creating visualizations...")
        plot_waveform(signal, args.sample_rate, duration=min(1.0, args.duration),
                     output_path=Path(args.output).stem + '_waveform.png')
        plot_frequency_spectrum(signal, args.sample_rate,
                               output_path=Path(args.output).stem + '_spectrum.png')
    
    print(f"✓ Audio generated successfully: {args.output}")


def list_items(args: argparse.Namespace) -> None:
    """List available generators and scales."""
    if args.type in ['generators', 'all']:
        print("\n=== Available Generators ===")
        generators = {
            'fourier': 'Fourier synthesis with harmonic series',
            'fibonacci_rhythm': 'Rhythm based on Fibonacci sequence',
            'lorenz': 'Lorenz attractor chaos modulation',
            'rossler': 'Rössler attractor chaos modulation',
            'logistic': 'Logistic map chaos modulation',
            'fractal': 'Fractal melody using L-systems',
            'prime': 'Melody based on prime numbers',
            'random_walk': 'Stochastic Brownian motion'
        }
        for name, desc in generators.items():
            print(f"  {name:20} - {desc}")
    
    if args.type in ['scales', 'all']:
        print("\n=== Available Scales ===")
        scales = {
            'chromatic': 'All 12 semitones',
            'major': 'Major scale (Ionian)',
            'minor': 'Natural minor scale (Aeolian)',
            'pentatonic_major': 'Major pentatonic (5 notes)',
            'pentatonic_minor': 'Minor pentatonic (5 notes)'
        }
        for name, desc in scales.items():
            print(f"  {name:20} - {desc}")
    
    print()


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'generate':
            if args.function:
                generate_from_function(args)
            elif args.generator:
                generate_from_generator(args)
        elif args.command == 'list':
            list_items(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
