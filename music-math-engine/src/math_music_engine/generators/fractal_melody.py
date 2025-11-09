"""
Fractal melody generator.

Generates melodies using L-systems and recursive fractal patterns.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np


class FractalMelody:
    """
    Melody generator using fractal and L-system algorithms.
    
    Mathematical basis: L-systems (Lindenmayer systems) and recursive patterns
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize fractal melody generator.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        
    def generate_lsystem(
        self,
        axiom: str,
        rules: Dict[str, str],
        iterations: int
    ) -> str:
        """
        Generate L-system string.
        
        Args:
            axiom: Starting string
            rules: Production rules (symbol -> replacement)
            iterations: Number of iterations
            
        Returns:
            Generated L-system string
        """
        current = axiom
        
        for _ in range(iterations):
            next_gen = ""
            for symbol in current:
                next_gen += rules.get(symbol, symbol)
            current = next_gen
        
        return current
    
    def lsystem_to_melody(
        self,
        lsystem_string: str,
        symbol_map: Optional[Dict[str, int]] = None,
        base_note: int = 60
    ) -> List[int]:
        """
        Convert L-system string to MIDI note sequence.
        
        Args:
            lsystem_string: L-system generated string
            symbol_map: Mapping from symbols to pitch intervals
                       If None, uses default mapping
            base_note: Base MIDI note
            
        Returns:
            List of MIDI note numbers
        """
        if symbol_map is None:
            # Default mapping: A=+2, B=-1, C=+3, etc.
            symbol_map = {
                'A': 2, 'B': -1, 'C': 3, 'D': -2,
                'E': 4, 'F': 1, 'G': -3, 'H': 5,
                '+': 7, '-': -7, '[': 0, ']': 0
            }
        
        melody = [base_note]
        current_note = base_note
        
        for symbol in lsystem_string:
            interval = symbol_map.get(symbol, 0)
            current_note += interval
            # Keep in MIDI range
            current_note = np.clip(current_note, 0, 127)
            melody.append(current_note)
        
        return melody
    
    def generate_cantor_rhythm(
        self,
        iterations: int,
        total_duration: float
    ) -> List[Tuple[float, float]]:
        """
        Generate rhythm based on Cantor set.
        
        Args:
            iterations: Number of Cantor set iterations
            total_duration: Total duration in seconds
            
        Returns:
            List of (start_time, duration) tuples
        """
        def cantor_set(start: float, end: float, depth: int) -> List[Tuple[float, float]]:
            if depth == 0:
                return [(start, end - start)]
            
            third = (end - start) / 3
            left = cantor_set(start, start + third, depth - 1)
            right = cantor_set(end - third, end, depth - 1)
            
            return left + right
        
        intervals = cantor_set(0, total_duration, iterations)
        return intervals
    
    def generate_recursive_melody(
        self,
        seed_pattern: List[int],
        recursion_depth: int,
        transformation: str = "mirror"
    ) -> List[int]:
        """
        Generate melody using recursive transformations.
        
        Args:
            seed_pattern: Initial melodic pattern (MIDI intervals)
            recursion_depth: Depth of recursion
            transformation: Type of transformation:
                          - "mirror": Mirror the pattern
                          - "invert": Invert intervals
                          - "retrograde": Reverse the pattern
                          - "expand": Multiply intervals by 2
            
        Returns:
            Recursive melodic pattern
        """
        def transform(pattern: List[int]) -> List[int]:
            if transformation == "mirror":
                return pattern + pattern[::-1]
            elif transformation == "invert":
                return pattern + [-x for x in pattern]
            elif transformation == "retrograde":
                return pattern + list(reversed(pattern))
            elif transformation == "expand":
                return pattern + [x * 2 for x in pattern]
            else:
                return pattern + pattern
        
        current_pattern = seed_pattern.copy()
        
        for _ in range(recursion_depth):
            current_pattern = transform(current_pattern)
        
        return current_pattern
    
    def generate_sierpinski_melody(
        self,
        iterations: int,
        base_interval: int = 2
    ) -> List[int]:
        """
        Generate melody based on Sierpinski triangle pattern.
        
        Args:
            iterations: Number of iterations
            base_interval: Base interval in semitones
            
        Returns:
            List of MIDI note intervals
        """
        # Generate Sierpinski triangle as binary pattern
        def sierpinski_row(n: int) -> List[int]:
            if n == 0:
                return [1]
            
            prev = sierpinski_row(n - 1)
            # Each row is previous XOR-ed with shifted version
            row = prev + [0] + prev
            return row
        
        # Convert to melody
        pattern = sierpinski_row(iterations)
        melody = [base_interval if x == 1 else 0 for x in pattern]
        
        return melody
    
    def generate_koch_curve_melody(
        self,
        iterations: int,
        intervals: List[int] = None
    ) -> List[int]:
        """
        Generate melody based on Koch curve (snowflake).
        
        Args:
            iterations: Number of Koch curve iterations
            intervals: Four intervals for Koch curve segments
                      Default: [2, 5, -5, 2]
            
        Returns:
            List of MIDI note intervals
        """
        if intervals is None:
            intervals = [2, 5, -5, 2]  # Up, up, down, up
        
        def koch_iteration(segment: List[int]) -> List[int]:
            result = []
            for interval in segment:
                # Replace each interval with Koch pattern
                scale = interval / sum(abs(x) for x in intervals)
                result.extend([int(x * scale) for x in intervals])
            return result
        
        melody = intervals.copy()
        
        for _ in range(iterations):
            melody = koch_iteration(melody)
        
        return melody
    
    def intervals_to_notes(
        self,
        intervals: List[int],
        base_note: int = 60
    ) -> List[int]:
        """
        Convert interval pattern to MIDI notes.
        
        Args:
            intervals: List of intervals in semitones
            base_note: Starting MIDI note
            
        Returns:
            List of MIDI note numbers
        """
        notes = [base_note]
        current = base_note
        
        for interval in intervals:
            current += interval
            current = np.clip(current, 0, 127)
            notes.append(current)
        
        return notes
    
    def generate_dragon_curve_melody(
        self,
        iterations: int,
        turn_interval: int = 3
    ) -> List[int]:
        """
        Generate melody based on dragon curve.
        
        Args:
            iterations: Number of iterations
            turn_interval: Interval for each turn (in semitones)
            
        Returns:
            List of MIDI note intervals
        """
        # Dragon curve: each iteration adds turns
        def dragon_sequence(n: int) -> List[int]:
            if n == 0:
                return [1]
            
            prev = dragon_sequence(n - 1)
            # Dragon curve: prev + [1] + reverse(invert(prev))
            inverted = [-x for x in prev]
            return prev + [1] + inverted[::-1]
        
        turns = dragon_sequence(iterations)
        # Convert turns to intervals
        melody = [turn_interval * turn for turn in turns]
        
        return melody
    
    def get_metadata(self, method: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get generator metadata.
        
        Args:
            method: Generation method used
            parameters: Method parameters
            
        Returns:
            Dictionary containing generator configuration
        """
        return {
            'generator': 'FractalMelody',
            'method': method,
            'parameters': parameters,
            'sample_rate': self.sample_rate,
            'mathematical_basis': 'L-systems and fractal recursion'
        }


# Predefined L-system rules
LSYSTEM_PRESETS = {
    'algae': {
        'axiom': 'A',
        'rules': {'A': 'AB', 'B': 'A'},
        'description': 'Fibonacci-like growth'
    },
    'binary_tree': {
        'axiom': '0',
        'rules': {'1': '11', '0': '1[0]0'},
        'description': 'Binary tree structure'
    },
    'cantor': {
        'axiom': 'A',
        'rules': {'A': 'ABA', 'B': 'BBB'},
        'description': 'Cantor set approximation'
    },
    'thue_morse': {
        'axiom': 'A',
        'rules': {'A': 'AB', 'B': 'BA'},
        'description': 'Thue-Morse sequence'
    }
}
