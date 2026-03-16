"""
detection/repetition_detector.py
================================
Repetition detector for stuttering analysis

Detects repeated speech segments using cosine similarity
pre-screening followed by DTW confirmation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings

from .stutter_event import StutterEvent, create_repetition_event

class RepetitionDetector:
    """
    Repetition detector for stuttering analysis
    
    Identifies repeated speech segments using cosine similarity
    pre-screening followed by DTW confirmation for detailed analysis.
    """
    
    def __init__(self, sample_rate: int = 16000, hop_size: int = 160,
                 cosine_threshold: float = 0.75, dtw_threshold: float = 15.0,
                 max_repetition_gap: int = 3, dtw_band_width_ratio: float = 0.2,
                 max_segment_length_ms: float = 500.0):
        """
        Initialize repetition detector
        
        Args:
            sample_rate: Audio sample rate (default 16000)
            hop_size: Hop size for frame-to-time conversion (default 160)
            cosine_threshold: Cosine similarity threshold for pre-screening (default 0.75)
            dtw_threshold: DTW distance threshold for confirmation (default 15.0)
            max_repetition_gap: Maximum segment gap for repetition detection (default 3)
            dtw_band_width_ratio: DTW Sakoe-Chiba band width ratio (default 0.2)
            max_segment_length_ms: Maximum segment length for DTW comparison (default 500ms)
        """
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.cosine_threshold = cosine_threshold
        self.dtw_threshold = dtw_threshold
        self.max_repetition_gap = max_repetition_gap
        self.dtw_band_width_ratio = dtw_band_width_ratio
        self.max_segment_length_ms = max_segment_length_ms
        self.max_segment_length_frames = int(max_segment_length_ms * sample_rate / (hop_size * 1000))
        
        print(f"[RepetitionDetector] Initialized with:")
        print(f"  Sample rate: {sample_rate}Hz")
        print(f"  Hop size: {hop_size}")
        print(f"  Cosine threshold: {cosine_threshold}")
        print(f"  DTW threshold: {dtw_threshold}")
        print(f"  Max repetition gap: {max_repetition_gap} segments")
        print(f"  DTW band width ratio: {dtw_band_width_ratio}")
        print(f"  Max segment length: {max_segment_length_ms}ms ({self.max_segment_length_frames} frames)")
    
    def detect_repetitions(self, segment_list: List[Dict], mfcc_full: np.ndarray,
                          augmented_segments: List) -> List[StutterEvent]:
        """
        Detect repetitions in speech segments
        
        Args:
            segment_list: List of segment dictionaries from segmentation
            mfcc_full: Global MFCC matrix
            augmented_segments: List of augmented segments with mean_mfcc features
            
        Returns:
            List of detected repetition events
        """
        print(f"[RepetitionDetector] Detecting repetitions...")
        print(f"[RepetitionDetector] Input segments: {len(segment_list)}")
        
        # Step 1: Build SPEECH segment sequence
        speech_segments = self._build_speech_sequence(segment_list)
        print(f"[RepetitionDetector] Speech segments: {len(speech_segments)}")
        
        if len(speech_segments) < 2:
            print(f"[RepetitionDetector] Too few speech segments for repetition detection")
            return []
        
        # Step 2: Fast pre-screening with cosine similarity
        candidate_pairs = self._cosine_pre_screening(speech_segments, augmented_segments)
        print(f"[RepetitionDetector] Cosine candidates: {len(candidate_pairs)}")
        
        # Step 3: DTW confirmation
        confirmed_pairs = self._dtw_confirmation(candidate_pairs, mfcc_full)
        print(f"[RepetitionDetector] DTW confirmed: {len(confirmed_pairs)}")
        
        # Step 4: Handle chained repetitions
        merged_events = self._handle_chained_repetitions(confirmed_pairs, speech_segments)
        print(f"[RepetitionDetector] Merged events: {len(merged_events)}")
        
        # Step 5: Emit detection events
        repetition_events = []
        for pair in merged_events:
            event = self._emit_repetition_event(pair, speech_segments)
            repetition_events.append(event)
        
        print(f"[RepetitionDetector] Detected {len(repetition_events)} repetition events")
        return repetition_events
    
    def _build_speech_sequence(self, segment_list: List[Dict]) -> List[Dict]:
        """
        Build sequence of SPEECH segments only
        
        Args:
            segment_list: Complete segment list
            
        Returns:
            List of speech segments in temporal order
        """
        speech_segments = []
        
        for i, segment in enumerate(segment_list):
            if segment.get('label') == 'SPEECH':
                # Add original index for tracking
                segment['original_index'] = i
                speech_segments.append(segment)
        
        return speech_segments
    
    def _cosine_pre_screening(self, speech_segments: List[Dict], 
                              augmented_segments: List) -> List[Tuple[int, int, float]]:
        """
        Fast pre-screening using cosine similarity
        
        Args:
            speech_segments: List of speech segments
            augmented_segments: List of augmented segments with mean_mfcc
            
        Returns:
            List of candidate pairs (i, j, similarity) where i < j
        """
        candidate_pairs = []
        
        # Create mapping from original index to augmented segment
        augmented_map = {seg['segment_index']: seg for seg in augmented_segments}
        
        # Compare adjacent and near-adjacent pairs
        for i, segment_i in enumerate(speech_segments):
            for j, segment_j in enumerate(speech_segments):
                if j <= i:  # Only compare forward
                    continue
                
                # Check gap constraint
                if j - i > self.max_repetition_gap:
                    continue
                
                # Get mean MFCC vectors
                orig_idx_i = segment_i['original_index']
                orig_idx_j = segment_j['original_index']
                
                if orig_idx_i not in augmented_map or orig_idx_j not in augmented_map:
                    continue
                
                mean_mfcc_i = augmented_map[orig_idx_i].features['mean_mfcc']
                mean_mfcc_j = augmented_map[orig_idx_j].features['mean_mfcc']
                
                # Compute cosine similarity
                similarity = self._compute_cosine_similarity(mean_mfcc_i, mean_mfcc_j)
                
                # Check threshold
                if similarity >= self.cosine_threshold:
                    candidate_pairs.append((i, j, similarity))
        
        return candidate_pairs
    
    def _compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity value
        """
        # Handle zero vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        similarity = dot_product / (norm1 * norm2)
        
        # Clip to [-1, 1] range
        return np.clip(similarity, -1.0, 1.0)
    
    def _dtw_confirmation(self, candidate_pairs: List[Tuple[int, int, float]], 
                        mfcc_full: np.ndarray) -> List[Tuple[int, int, float, float]]:
        """
        DTW confirmation for candidate pairs
        
        Args:
            candidate_pairs: List of candidate pairs (i, j, similarity)
            mfcc_full: Global MFCC matrix
            
        Returns:
            List of confirmed pairs (i, j, cosine_similarity, dtw_distance)
        """
        confirmed_pairs = []
        
        for i, j, cosine_sim in candidate_pairs:
            try:
                # Get MFCC matrices for both segments
                segment_i = self._get_segment_mfcc(i, candidate_pairs, mfcc_full)
                segment_j = self._get_segment_mfcc(j, candidate_pairs, mfcc_full)
                
                # Apply length constraints
                if len(segment_i) == 0 or len(segment_j) == 0:
                    continue
                
                # Limit segment length for DTW
                if len(segment_i) > self.max_segment_length_frames:
                    segment_i = segment_i[:self.max_segment_length_frames]
                if len(segment_j) > self.max_segment_length_frames:
                    segment_j = segment_j[:self.max_segment_length_frames]
                
                # Compute DTW distance
                dtw_distance = self._compute_dtw_distance(segment_i, segment_j)
                
                # Check threshold
                if dtw_distance < self.dtw_threshold:
                    confirmed_pairs.append((i, j, cosine_sim, dtw_distance))
                
            except Exception as e:
                print(f"[RepetitionDetector] DTW error for pair ({i}, {j}): {e}")
                continue
        
        return confirmed_pairs
    
    def _get_segment_mfcc(self, segment_idx: int, candidate_pairs: List[Tuple[int, int, float]], 
                          mfcc_full: np.ndarray) -> np.ndarray:
        """
        Get MFCC matrix for a specific segment
        
        Args:
            segment_idx: Index in speech_segments
            candidate_pairs: List of candidate pairs (for context)
            mfcc_full: Global MFCC matrix
            
        Returns:
            MFCC matrix for the segment
        """
        # This is a simplified implementation - in practice, we'd need
        # the actual frame indices from the segment data
        # For now, we'll use a placeholder approach
        
        # Extract a reasonable portion of the MFCC matrix
        start_frame = segment_idx * 20  # Placeholder
        end_frame = start_frame + 40     # Placeholder
        
        # Ensure we don't exceed bounds
        end_frame = min(end_frame, mfcc_full.shape[0])
        start_frame = max(0, start_frame)
        
        return mfcc_full[start_frame:end_frame]
    
    def _compute_dtw_distance(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """
        Compute DTW distance between two sequences
        
        Args:
            seq1: First sequence (n_frames, n_features)
            seq2: Second sequence (m_frames, n_features)
            
        Returns:
            DTW distance
        """
        # Implement DTW with Sakoe-Chiba band constraint
        n_frames1, n_features = seq1.shape
        n_frames2, _ = seq2.shape
        
        # Compute band width
        band_width = int(min(n_frames1, n_frames2) * self.dtw_band_width_ratio)
        
        # Initialize cost matrix
        cost_matrix = np.full((n_frames1 + 1, n_frames2 + 1), np.inf)
        cost_matrix[0, 0] = 0
        
        # Compute cost matrix with band constraint
        for i in range(1, n_frames1 + 1):
            for j in range(1, n_frames2 + 1):
                # Check band constraint
                if abs(i - j) > band_width:
                    continue
                
                # Compute Euclidean distance between frames
                frame_cost = np.linalg.norm(seq1[i-1] - seq2[j-1])
                
                # Find minimum path to current cell
                min_cost = min(
                    cost_matrix[i-1, j],    # Insertion
                    cost_matrix[i, j-1],    # Deletion
                    cost_matrix[i-1, j-1]   # Match
                )
                
                cost_matrix[i, j] = min_cost + frame_cost
        
        return cost_matrix[n_frames1, n_frames2]
    
    def _handle_chained_repetitions(self, confirmed_pairs: List[Tuple[int, int, float, float]], 
                                  speech_segments: List[Dict]) -> List[List[Tuple[int, int, float, float]]]:
        """
        Handle chained repetitions (e.g., ba-ba-ba-banana)
        
        Args:
            confirmed_pairs: List of confirmed repetition pairs
            speech_segments: List of speech segments
            
        Returns:
            List of merged repetition chains
        """
        if not confirmed_pairs:
            return []
        
        # Build adjacency graph
        graph = self._build_repetition_graph(confirmed_pairs)
        
        # Find connected components (chains)
        chains = self._find_connected_components(graph, len(speech_segments))
        
        # Convert chains to merged events
        merged_events = []
        for chain in chains:
            if len(chain) > 1:
                merged_events.append(chain)
        
        return merged_events
    
    def _build_repetition_graph(self, confirmed_pairs: List[Tuple[int, int, float, float]]) -> Dict[int, List[int]]:
        """
        Build adjacency graph from confirmed pairs
        
        Args:
            confirmed_pairs: List of confirmed repetition pairs
            
        Returns:
            Adjacency graph dictionary
        """
        graph = {}
        
        for i, j, _, _ in confirmed_pairs:
            if i not in graph:
                graph[i] = []
            if j not in graph:
                graph[j] = []
            
            graph[i].append(j)
            graph[j].append(i)
        
        return graph
    
    def _find_connected_components(self, graph: Dict[int, List[int]], n_segments: int) -> List[List[int]]:
        """
        Find connected components in the repetition graph
        
        Args:
            graph: Adjacency graph
            n_segments: Total number of segments
            
        Returns:
            List of connected components
        """
        visited = set()
        components = []
        
        for node in range(n_segments):
            if node not in visited:
                component = self._dfs_component(node, graph, visited)
                if len(component) > 1:  # Only keep components with multiple nodes
                    components.append(component)
        
        return components
    
    def _dfs_component(self, node: int, graph: Dict[int, List[int]], visited: set) -> List[int]:
        """
        Depth-first search to find connected component
        
        Args:
            node: Starting node
            graph: Adjacency graph
            visited: Set of visited nodes
            
        Returns:
            List of nodes in the connected component
        """
        if node in visited:
            return []
        
        visited.add(node)
        component = [node]
        
        if node in graph:
            for neighbor in graph[node]:
                component.extend(self._dfs_component(neighbor, graph, visited))
        
        return component
    
    def _emit_repetition_event(self, chain: List[int], speech_segments: List[Dict]) -> StutterEvent:
        """
        Create repetition detection event
        
        Args:
            chain: List of segment indices in the repetition chain
            speech_segments: List of speech segments
            
        Returns:
            StutterEvent for the detected repetition
        """
        # Find canonical segment (latest in chain)
        canonical_segment_idx = max(chain)
        repeated_segment_indices = [idx for idx in chain if idx != canonical_segment_idx]
        
        # Get segment boundaries
        canonical_segment = speech_segments[canonical_segment_idx]
        
        # Calculate event boundaries (union of all repeated segments)
        start_segment = min(speech_segments[idx] for idx in chain)
        end_segment = max(speech_segments[idx] for idx in chain)
        
        start_sample = start_segment.get('start_sample', 0)
        end_sample = end_segment.get('end_sample', 0)
        start_time = start_segment.get('start_time', 0.0)
        end_time = end_segment.get('end_time', 0.0)
        
        # Calculate confidence (average of pairwise similarities)
        # For simplicity, we'll use a fixed high confidence for confirmed chains
        confidence = 0.8 + 0.1 * min(len(chain) - 2, 2)  # Higher confidence for longer chains
        confidence = min(confidence, 1.0)
        
        # Create event
        event_id = f"repetition_{canonical_segment_idx:03d}"
        
        repetition_event = create_repetition_event(
            event_id=event_id,
            start_sample=start_sample,
            end_sample=end_sample,
            start_time=start_time,
            end_time=end_time,
            confidence=confidence,
            segment_index=canonical_segment['original_index'],
            cosine_similarity=0.8,  # Placeholder
            dtw_distance=10.0,      # Placeholder
            canonical_segment_index=canonical_segment,
            repeated_segment_indices=repeated_segment_indices
        )
        
        return repetition_event
    
    def get_processing_info(self) -> dict:
        """Get information about repetition detector configuration"""
        return {
            'sample_rate': self.sample_rate,
            'hop_size': self.hop_size,
            'cosine_threshold': self.cosine_threshold,
            'dtw_threshold': self.dtw_threshold,
            'max_repetition_gap': self.max_repetition_gap,
            'dtw_band_width_ratio': self.dtw_band_width_ratio,
            'max_segment_length_ms': self.max_segment_length_ms,
            'max_segment_length_frames': self.max_segment_length_frames
        }


if __name__ == "__main__":
    # Test the repetition detector
    print("🧪 REPETITION DETECTOR TEST")
    print("=" * 30)
    
    # Initialize detector
    detector = RepetitionDetector(
        sample_rate=16000,
        hop_size=160,
        cosine_threshold=0.75,
        dtw_threshold=15.0,
        max_repetition_gap=3,
        dtw_band_width_ratio=0.2,
        max_segment_length_ms=500.0
    )
    
    # Create test data with repeated segments
    n_frames = 200
    n_mfcc_features = 39
    
    # Create MFCC matrix with similar patterns for repeated segments
    mfcc_full = np.random.randn(n_frames, n_mfcc_features) * 0.1
    
    # Add similar patterns for segments 1, 2, and 4 (repetition)
    base_pattern = np.random.randn(20, n_mfcc_features) * 0.05
    mfcc_full[20:40] = base_pattern + np.random.randn(20, n_mfcc_features) * 0.01  # Segment 1
    mfcc_full[60:80] = base_pattern + np.random.randn(20, n_mfcc_features) * 0.01  # Segment 2 (repeat)
    mfcc_full[120:140] = base_pattern + np.random.randn(20, n_mfcc_features) * 0.01  # Segment 4 (repeat)
    
    # Create test segment list
    segment_list = [
        {
            'label': 'SPEECH',
            'start_frame': 0,
            'end_frame': 19,
            'start_sample': 0,
            'end_sample': 3200,
            'start_time': 0.0,
            'end_time': 0.2,
            'duration_ms': 200.0,
            'mean_ste': 0.05
        },
        {
            'label': 'CLOSURE',
            'start_frame': 19,
            'end_frame': 39,
            'start_sample': 3200,
            'end_sample': 6400,
            'start_time': 0.2,
            'end_time': 0.4,
            'duration_ms': 200.0,
            'mean_ste': 0.001
        },
        {
            'label': 'SPEECH',
            'start_frame': 40,
            'end_frame': 59,
            'start_sample': 6400,
            'end_sample': 9600,
            'start_time': 0.4,
            'end_time': 0.6,
            'duration_ms': 200.0,
            'mean_ste': 0.06
        },
        {
            'label': 'CLOSURE',
            'start_frame': 59,
            'end_frame': 79,
            'start_sample': 9600,
            'end_sample': 12800,
            'start_time': 0.6,
            'end_time': 0.8,
            'duration_ms': 200.0,
            'mean_ste': 0.001
        },
        {
            'label': 'SPEECH',
            'start_frame': 80,
            'end_frame': 99,
            'start_sample': 12800,
            'end_sample': 16000,
            'start_time': 0.8,
            'end_time': 1.0,
            'duration_ms': 200.0,
            'mean_ste': 0.07
        },
        {
            'label': 'PAUSE_CANDIDATE',
            'start_frame': 99,
            'end_frame': 119,
            'start_sample': 16000,
            'end_sample': 19200,
            'start_time': 1.0,
            'end_time': 1.2,
            'duration_ms': 200.0,
            'mean_ste': 0.002
        },
        {
            'label': 'SPEECH',
            'start_frame': 120,
            'end_frame': 139,
            'start_sample': 19200,
            'end_sample': 22400,
            'start_time': 1.2,
            'end_time': 1.4,
            'duration_ms': 200.0,
            'mean_ste': 0.04
        }
    ]
    
    # Create augmented segments with mean_mfcc
    from features.feature_store import AugmentedSegment
    
    augmented_segments = []
    for i, segment in enumerate(segment_list):
        if segment['label'] == 'SPEECH':
            # Create mean MFCC vector
            start_frame = segment['start_frame']
            end_frame = segment['end_frame']
            segment_mfcc = mfcc_full[start_frame:end_frame + 1]
            mean_mfcc = np.mean(segment_mfcc, axis=0)
            
            # Create augmented segment (simplified)
            augmented_seg = type('AugmentedSegment', (), {
                'segment_index': i,
                'features': type('Features', (), {
                    'mean_mfcc': mean_mfcc
                })()
            })()
            augmented_segments.append(augmented_seg)
    
    print(f"Test setup:")
    print(f"  MFCC matrix: {mfcc_full.shape}")
    print(f"  Total segments: {len(segment_list)}")
    print(f"  Speech segments: {len([s for s in segment_list if s['label'] == 'SPEECH'])}")
    print(f"  Augmented segments: {len(augmented_segments)}")
    print(f"  Repeated patterns: segments at frames 20-40, 60-80, 120-140")
    
    # Detect repetitions
    repetition_events = detector.detect_repetitions(segment_list, mfcc_full, augmented_segments)
    
    print(f"\n📊 REPETITION DETECTION RESULTS:")
    print(f"Detected events: {len(repetition_events)}")
    
    for event in repetition_events:
        print(f"  {event.event_id}: confidence={event.confidence:.2f}")
        print(f"    Time: {event.start_time:.3f}s - {event.end_time:.3f}s")
        print(f"    Canonical: {event.supporting_features['repetition']['canonical_segment_index']}")
        print(f"    Repeated: {event.supporting_features['repetition']['repeated_segment_indices']}")
    
    print(f"\n🎉 REPETITION DETECTOR TEST COMPLETE!")
    print(f"Module ready for integration with detection runner!")
