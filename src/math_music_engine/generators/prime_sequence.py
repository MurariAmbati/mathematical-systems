"""
Prime sequence generator.

Generates musical patterns based on prime numbers and prime number properties.
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np


class PrimeSequence:
    """
    Generator based on prime numbers and their properties.
    
    Mathematical basis: Prime numbers, prime gaps, prime residues
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize prime sequence generator.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self._prime_cache: List[int] = []
        
    def generate_primes(self, n: int) -> List[int]:
        """
        Generate first n prime numbers using Sieve of Eratosthenes.
        
        Args:
            n: Number of primes to generate
            
        Returns:
            List of prime numbers
        """
        if n <= 0:
            return []
        
        # Use cache if available
        if len(self._prime_cache) >= n:
            return self._prime_cache[:n]
        
        # Estimate upper bound for nth prime
        if n < 6:
            limit = 15
        else:
            limit = int(n * (np.log(n) + np.log(np.log(n)))) + 100
        
        # Sieve of Eratosthenes
        sieve = np.ones(limit, dtype=bool)
        sieve[0:2] = False
        
        for i in range(2, int(np.sqrt(limit)) + 1):
            if sieve[i]:
                sieve[i*i::i] = False
        
        primes = np.where(sieve)[0].tolist()
        self._prime_cache = primes
        
        return primes[:n]
    
    def generate_prime_gaps(self, n: int) -> List[int]:
        """
        Generate gaps between consecutive primes.
        
        Args:
            n: Number of gaps to generate
            
        Returns:
            List of prime gaps
        """
        primes = self.generate_primes(n + 1)
        gaps = [primes[i+1] - primes[i] for i in range(n)]
        return gaps
    
    def prime_melody(
        self,
        num_notes: int,
        base_note: int = 60,
        scale_factor: float = 0.5,
        method: str = "direct"
    ) -> List[int]:
        """
        Generate melody from prime numbers.
        
        Args:
            num_notes: Number of notes to generate
            base_note: Base MIDI note
            scale_factor: Scaling factor for prime-to-interval conversion
            method: Method for converting primes to notes:
                   - "direct": Use primes directly as intervals
                   - "gaps": Use prime gaps as intervals
                   - "residues": Use prime residues modulo 12
            
        Returns:
            List of MIDI note numbers
        """
        if method == "direct":
            primes = self.generate_primes(num_notes)
            intervals = [int(p * scale_factor) for p in primes]
            
        elif method == "gaps":
            gaps = self.generate_prime_gaps(num_notes)
            intervals = gaps
            
        elif method == "residues":
            primes = self.generate_primes(num_notes)
            intervals = [p % 12 for p in primes]
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert intervals to notes
        notes = [base_note]
        current_note = base_note
        
        for interval in intervals:
            current_note = base_note + (interval % 24)  # Keep within 2 octaves
            notes.append(current_note)
        
        return notes
    
    def prime_rhythm(
        self,
        num_beats: int,
        subdivision: int = 16
    ) -> np.ndarray:
        """
        Generate rhythm pattern where primes indicate beats.
        
        Args:
            num_beats: Number of beats
            subdivision: Subdivision per beat
            
        Returns:
            Binary rhythm pattern
        """
        total_slots = num_beats * subdivision
        pattern = np.zeros(total_slots, dtype=int)
        
        # Generate primes up to total_slots
        primes = []
        for p in self.generate_primes(total_slots):
            if p < total_slots:
                primes.append(p)
            else:
                break
        
        # Place beats at prime positions
        for prime in primes:
            if prime < total_slots:
                pattern[prime] = 1
        
        return pattern
    
    def prime_scale_degrees(
        self,
        num_notes: int,
        scale_size: int = 7
    ) -> List[int]:
        """
        Map primes to scale degrees.
        
        Args:
            num_notes: Number of notes
            scale_size: Size of scale (e.g., 7 for diatonic)
            
        Returns:
            List of scale degrees (0-indexed)
        """
        primes = self.generate_primes(num_notes)
        scale_degrees = [p % scale_size for p in primes]
        return scale_degrees
    
    def goldbach_melody(
        self,
        even_numbers: List[int],
        base_note: int = 60
    ) -> List[Tuple[int, int]]:
        """
        Generate note pairs using Goldbach's conjecture.
        
        Goldbach's conjecture: Every even integer > 2 can be expressed
        as the sum of two primes.
        
        Args:
            even_numbers: List of even numbers to decompose
            base_note: Base MIDI note
            
        Returns:
            List of note pairs (p1, p2) where p1 + p2 = even_number
        """
        def find_goldbach_pair(n: int) -> Tuple[int, int]:
            """Find a pair of primes that sum to n."""
            if n <= 2 or n % 2 != 0:
                return (0, 0)
            
            primes = self.generate_primes(n)
            prime_set = set(primes)
            
            for p in primes:
                if p > n:
                    break
                complement = n - p
                if complement in prime_set:
                    return (p, complement)
            
            return (0, 0)
        
        melody_pairs = []
        for num in even_numbers:
            p1, p2 = find_goldbach_pair(num)
            note1 = base_note + (p1 % 24)
            note2 = base_note + (p2 % 24)
            melody_pairs.append((note1, note2))
        
        return melody_pairs
    
    def twin_primes_pattern(
        self,
        num_pairs: int
    ) -> List[Tuple[int, int]]:
        """
        Generate pattern from twin primes (primes differing by 2).
        
        Args:
            num_pairs: Number of twin prime pairs to find
            
        Returns:
            List of twin prime pairs
        """
        twin_pairs = []
        primes = self.generate_primes(num_pairs * 10)  # Generate extra to find twins
        
        for i in range(len(primes) - 1):
            if primes[i+1] - primes[i] == 2:
                twin_pairs.append((primes[i], primes[i+1]))
                if len(twin_pairs) >= num_pairs:
                    break
        
        return twin_pairs
    
    def prime_frequency_ratios(
        self,
        num_ratios: int,
        base_freq: float = 440.0
    ) -> List[float]:
        """
        Generate frequency ratios based on prime numbers.
        
        Args:
            num_ratios: Number of frequencies to generate
            base_freq: Base frequency in Hz
            
        Returns:
            List of frequencies
        """
        primes = self.generate_primes(num_ratios)
        
        # Use primes as frequency multipliers
        frequencies = [base_freq * (p / primes[0]) for p in primes]
        
        return frequencies
    
    def ulam_spiral_melody(
        self,
        size: int,
        base_note: int = 60
    ) -> List[int]:
        """
        Generate melody from Ulam spiral (prime number spiral).
        
        Args:
            size: Size of spiral (will generate size^2 numbers)
            base_note: Base MIDI note
            
        Returns:
            List of MIDI notes (primes get higher pitch)
        """
        # Generate Ulam spiral
        spiral_size = size * size
        spiral = []
        
        # Create spiral pattern
        x, y = size // 2, size // 2
        dx, dy = 0, -1
        
        for i in range(1, spiral_size + 1):
            spiral.append(i)
            
            if x == y or (x < 0 and x == -y) or (x > 0 and x == 1 - y):
                dx, dy = -dy, dx
            
            x, y = x + dx, y + dy
        
        # Check which numbers are prime
        max_num = spiral_size
        primes_set = set(self.generate_primes(max_num))
        
        # Convert to melody
        melody = []
        for num in spiral:
            if num in primes_set:
                note = base_note + 7  # Perfect fifth above
            else:
                note = base_note
            melody.append(note)
        
        return melody
    
    def prime_modulation_signal(
        self,
        duration: float,
        prime_density: int = 10
    ) -> np.ndarray:
        """
        Generate modulation signal with impulses at prime-based positions.
        
        Args:
            duration: Duration in seconds
            prime_density: Number of primes per second
            
        Returns:
            Modulation signal
        """
        num_samples = int(duration * self.sample_rate)
        signal = np.zeros(num_samples)
        
        num_primes = int(duration * prime_density)
        primes = self.generate_primes(num_primes)
        
        for prime in primes:
            # Place impulse at position determined by prime
            position = (prime * num_samples) // (primes[-1] + 1)
            if position < num_samples:
                signal[position] = 1.0
        
        return signal
    
    def get_metadata(self, method: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get generator metadata.
        
        Args:
            method: Generation method
            parameters: Method parameters
            
        Returns:
            Dictionary containing generator configuration
        """
        return {
            'generator': 'PrimeSequence',
            'method': method,
            'parameters': parameters,
            'sample_rate': self.sample_rate,
            'mathematical_basis': 'Prime numbers and prime number theory'
        }
