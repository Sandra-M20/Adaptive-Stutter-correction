"""
professional_debug.py
=====================
Professional comprehensive testing and debugging system
"""

import os
import sys
import numpy as np
import soundfile as sf
import time
import traceback
from typing import Dict, List, Tuple, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class ProfessionalDebugger:
    """
    Professional debugging system for comprehensive analysis and fixes
    """
    
    def __init__(self):
        self.test_results = {}
        self.issues_found = []
        self.fixes_applied = []
        self.performance_metrics = {}
        
    def run_comprehensive_analysis(self) -> Dict:
        """
        Run complete professional analysis of the entire system
        """
        print("🔬 PROFESSIONAL COMPREHENSIVE SYSTEM ANALYSIS")
        print("=" * 60)
        print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        analysis_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_status': 'UNKNOWN',
            'critical_issues': [],
            'performance_issues': [],
            'functional_issues': [],
            'recommendations': [],
            'fixes_needed': []
        }
        
        # Phase 1: System Architecture Analysis
        print("\n🏗️ PHASE 1: SYSTEM ARCHITECTURE ANALYSIS")
        print("-" * 50)
        arch_issues = self.analyze_system_architecture()
        analysis_results['functional_issues'].extend(arch_issues)
        
        # Phase 2: Component Testing
        print("\n🧪 PHASE 2: COMPONENT TESTING")
        print("-" * 50)
        component_results = self.test_all_components()
        analysis_results['functional_issues'].extend(component_results['issues'])
        
        # Phase 3: Integration Testing
        print("\n🔗 PHASE 3: INTEGRATION TESTING")
        print("-" * 50)
        integration_results = self.test_integration()
        analysis_results['functional_issues'].extend(integration_results['issues'])
        
        # Phase 4: Performance Analysis
        print("\n⚡ PHASE 4: PERFORMANCE ANALYSIS")
        print("-" * 50)
        perf_results = self.analyze_performance()
        analysis_results['performance_issues'] = perf_results['issues']
        
        # Phase 5: Real-world Testing
        print("\n🌍 PHASE 5: REAL-WORLD TESTING")
        print("-" * 50)
        realworld_results = self.test_real_world_scenarios()
        analysis_results['functional_issues'].extend(realworld_results['issues'])
        
        # Phase 6: Critical Issue Assessment
        print("\n🚨 PHASE 6: CRITICAL ISSUE ASSESSMENT")
        print("-" * 50)
        critical_issues = self.assess_critical_issues(analysis_results)
        analysis_results['critical_issues'] = critical_issues
        
        # Phase 7: Generate Fixes
        print("\n🔧 PHASE 7: GENERATING FIXES")
        print("-" * 50)
        fixes = self.generate_comprehensive_fixes(analysis_results)
        analysis_results['fixes_needed'] = fixes
        
        # Final Assessment
        total_issues = len(analysis_results['critical_issues']) + len(analysis_results['functional_issues'])
        if total_issues == 0:
            analysis_results['system_status'] = 'HEALTHY'
        elif len(analysis_results['critical_issues']) == 0:
            analysis_results['system_status'] = 'NEEDS_MINOR_FIXES'
        else:
            analysis_results['system_status'] = 'NEEDS_MAJOR_FIXES'
        
        self.print_final_summary(analysis_results)
        
        return analysis_results
    
    def analyze_system_architecture(self) -> List[Dict]:
        """Analyze system architecture for design issues"""
        issues = []
        
        print("  🔍 Checking file structure...")
        
        # Check core files exist
        required_files = [
            'pipeline.py', 'config.py', 'preprocessing.py', 'segmentation.py',
            'pause_corrector.py', 'prolongation_corrector.py', 'repetition_corrector.py',
            'speech_reconstructor.py', 'audio_enhancer.py'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            issues.append({
                'type': 'CRITICAL',
                'component': 'Architecture',
                'issue': f'Missing core files: {missing_files}',
                'impact': 'System cannot function'
            })
        
        # Check for duplicate/conflicting implementations
        pipeline_files = ['pipeline.py', 'fixed_pipeline.py', 'balanced_pipeline.py', 'conservative_pipeline.py']
        existing_pipelines = [f for f in pipeline_files if os.path.exists(f)]
        
        if len(existing_pipelines) > 1:
            issues.append({
                'type': 'WARNING',
                'component': 'Architecture',
                'issue': f'Multiple pipeline implementations found: {existing_pipelines}',
                'impact': 'Confusion about which pipeline to use'
            })
        
        print(f"  ✅ Found {len(required_files) - len(missing_files)}/{len(required_files)} core files")
        print(f"  ⚠️  Found {len(existing_pipelines)} pipeline implementations")
        
        return issues
    
    def test_all_components(self) -> Dict:
        """Test all individual components"""
        results = {'issues': [], 'tested': [], 'failed': []}
        
        components = [
            ('Config', 'config', 'TARGET_SR'),
            ('Preprocessing', 'preprocessing', 'AudioPreprocessor'),
            ('Segmentation', 'segmentation', 'SpeechSegmenter'),
            ('Pause Corrector', 'pause_corrector', 'PauseCorrector'),
            ('Prolongation Corrector', 'prolongation_corrector', 'ProlongationCorrector'),
            ('Repetition Corrector', 'repetition_corrector', 'RepetitionCorrector'),
            ('Speech Reconstructor', 'speech_reconstructor', 'SpeechReconstructor'),
            ('Audio Enhancer', 'audio_enhancer', 'AudioEnhancer'),
            ('Silent Stutter Detector', 'silent_stutter_detector', 'SilentStutterDetector'),
            ('AI Performance Monitor', 'ai_performance_monitor', 'AIPerformanceMonitor')
        ]
        
        for name, module_name, class_name in components:
            print(f"  🧪 Testing {name}...")
            
            try:
                module = __import__(module_name)
                cls = getattr(module, class_name)
                
                # Test instantiation
                if name == 'Config':
                    # Config is special - just test values
                    sr = getattr(module, 'TARGET_SR', None)
                    if sr is None or sr <= 0:
                        raise ValueError("Invalid TARGET_SR")
                else:
                    # Test class instantiation
                    if name == 'Preprocessing':
                        instance = cls(noise_reduce=False)
                    elif name == 'Segmentation':
                        instance = cls(sr=22050)
                    elif name in ['Pause Corrector', 'Prolongation Corrector']:
                        instance = cls(sr=22050)
                    elif name == 'Repetition Corrector':
                        instance = cls(sr=22050)
                    elif name == 'Silent Stutter Detector':
                        instance = cls(sr=22050)
                    elif name == 'Audio Enhancer':
                        instance = cls()
                    elif name == 'Speech Reconstructor':
                        instance = cls()
                    elif name == 'AI Performance Monitor':
                        instance = cls()
                
                results['tested'].append(name)
                print(f"    ✅ {name} OK")
                
            except Exception as e:
                error_msg = f"{name} failed: {str(e)}"
                results['failed'].append(name)
                results['issues'].append({
                    'type': 'ERROR',
                    'component': name,
                    'issue': error_msg,
                    'traceback': traceback.format_exc()
                })
                print(f"    ❌ {name} FAILED: {str(e)}")
        
        print(f"  📊 Component Test Results: {len(results['tested'])} passed, {len(results['failed'])} failed")
        return results
    
    def test_integration(self) -> Dict:
        """Test system integration"""
        results = {'issues': [], 'tested': [], 'failed': []}
        
        print("  🔗 Testing pipeline integration...")
        
        # Test different pipeline implementations
        pipelines = [
            ('Main Pipeline', 'pipeline', 'StutterCorrectionPipeline'),
            ('Conservative Pipeline', 'conservative_pipeline', 'ConservativeStutterCorrectionPipeline')
        ]
        
        for name, module_name, class_name in pipelines:
            if not os.path.exists(f"{module_name}.py"):
                continue
                
            print(f"    🧪 Testing {name}...")
            
            try:
                module = __import__(module_name)
                cls = getattr(module, class_name)
                
                # Test instantiation
                if name == 'Main Pipeline':
                    instance = cls(
                        use_adaptive=False,
                        use_repetition=True,
                        use_enhancer=False,
                        transcribe=False
                    )
                else:
                    instance = cls()
                
                results['tested'].append(name)
                print(f"      ✅ {name} instantiation OK")
                
                # Test with synthetic audio if possible
                try:
                    # Create synthetic test audio
                    sr = 22050
                    duration = 2.0
                    t = np.linspace(0, duration, int(sr * duration))
                    test_signal = 0.5 * np.sin(2 * np.pi * 440 * t)
                    
                    # Save test file
                    test_file = "integration_test.wav"
                    sf.write(test_file, test_signal, sr)
                    
                    # Test processing
                    if name == 'Main Pipeline':
                        result = instance.run(test_file, output_dir="test_output")
                    else:
                        result = instance.correct(test_file, "integration_output.wav")
                    
                    print(f"      ✅ {name} processing OK")
                    
                    # Clean up
                    if os.path.exists(test_file):
                        os.remove(test_file)
                        
                except Exception as process_error:
                    results['issues'].append({
                        'type': 'WARNING',
                        'component': f"{name} Processing",
                        'issue': f"Processing test failed: {str(process_error)}",
                        'traceback': traceback.format_exc()
                    })
                    print(f"      ⚠️  {name} processing test failed: {str(process_error)}")
                
            except Exception as e:
                error_msg = f"{name} integration failed: {str(e)}"
                results['failed'].append(name)
                results['issues'].append({
                    'type': 'ERROR',
                    'component': name,
                    'issue': error_msg,
                    'traceback': traceback.format_exc()
                })
                print(f"      ❌ {name} integration FAILED: {str(e)}")
        
        print(f"  📊 Integration Test Results: {len(results['tested'])} passed, {len(results['failed'])} failed")
        return results
    
    def analyze_performance(self) -> Dict:
        """Analyze system performance"""
        results = {'issues': [], 'metrics': {}}
        
        print("  ⚡ Analyzing performance...")
        
        # Test processing speed
        try:
            from conservative_pipeline import ConservativeStutterCorrectionPipeline
            
            pipeline = ConservativeStutterCorrectionPipeline()
            
            # Create test audio of different sizes
            test_durations = [1.0, 3.0, 5.0]  # seconds
            rtf_scores = []
            
            for duration in test_durations:
                sr = 22050
                t = np.linspace(0, duration, int(sr * duration))
                test_signal = 0.5 * np.sin(2 * np.pi * 440 * t)
                
                test_file = f"perf_test_{duration}s.wav"
                sf.write(test_file, test_signal, sr)
                
                start_time = time.time()
                result = pipeline.correct(test_file, f"perf_output_{duration}s.wav")
                processing_time = time.time() - start_time
                
                rtf = processing_time / duration
                rtf_scores.append(rtf)
                
                print(f"    📊 {duration}s audio: RTF = {rtf:.2f}")
                
                # Clean up
                if os.path.exists(test_file):
                    os.remove(test_file)
                if os.path.exists(f"perf_output_{duration}s.wav"):
                    os.remove(f"perf_output_{duration}s.wav")
            
            avg_rtf = np.mean(rtf_scores)
            results['metrics']['average_rtf'] = avg_rtf
            
            if avg_rtf > 2.0:
                results['issues'].append({
                    'type': 'PERFORMANCE',
                    'component': 'Processing Speed',
                    'issue': f"Average RTF {avg_rtf:.2f} is too slow (should be < 2.0)",
                    'impact': 'Real-time processing may not be feasible'
                })
            else:
                print(f"    ✅ Performance OK: Average RTF = {avg_rtf:.2f}")
                
        except Exception as e:
            results['issues'].append({
                'type': 'ERROR',
                'component': 'Performance Analysis',
                'issue': f"Performance testing failed: {str(e)}",
                'traceback': traceback.format_exc()
            })
        
        return results
    
    def test_real_world_scenarios(self) -> Dict:
        """Test with real-world scenarios"""
        results = {'issues': [], 'scenarios_tested': 0}
        
        print("  🌍 Testing real-world scenarios...")
        
        # Find test audio files
        test_files = []
        for file in ['test_input.wav', '_selftest_input.wav', 'output/_test_stutter_original.wav']:
            if os.path.exists(file):
                test_files.append(file)
        
        if not test_files:
            results['issues'].append({
                'type': 'WARNING',
                'component': 'Real-world Testing',
                'issue': 'No test audio files found for real-world testing',
                'impact': 'Cannot validate with actual stuttering audio'
            })
            return results
        
        # Test with conservative pipeline (most stable)
        try:
            from conservative_pipeline import ConservativeStutterCorrectionPipeline
            pipeline = ConservativeStutterCorrectionPipeline()
            
            for i, test_file in enumerate(test_files):
                print(f"    🧪 Testing scenario {i+1}: {test_file}")
                
                try:
                    # Load original for comparison
                    original_signal, sr = sf.read(test_file)
                    if len(original_signal.shape) > 1:
                        original_signal = np.mean(original_signal, axis=1)
                    
                    original_duration = len(original_signal) / sr
                    original_energy = np.sum(original_signal ** 2)
                    
                    # Process
                    output_file = f"realworld_test_{i+1}.wav"
                    result = pipeline.correct(test_file, output_file)
                    
                    # Validate output
                    if os.path.exists(output_file):
                        corrected_signal, sr_corrected = sf.read(output_file)
                        if len(corrected_signal.shape) > 1:
                            corrected_signal = np.mean(corrected_signal, axis=1)
                        
                        corrected_duration = len(corrected_signal) / sr_corrected
                        corrected_energy = np.sum(corrected_signal ** 2)
                        
                        # Validation checks
                        issues = []
                        
                        # Duration should be reasonable
                        reduction = (1 - corrected_duration / original_duration) * 100
                        if reduction > 30:
                            issues.append(f"Excessive duration reduction: {reduction:.1f}%")
                        elif reduction < 0:
                            issues.append(f"Duration increased: {-reduction:.1f}%")
                        
                        # Energy should be preserved
                        energy_ratio = corrected_energy / original_energy
                        if energy_ratio < 0.3:
                            issues.append(f"Excessive energy loss: {energy_ratio*100:.0f}%")
                        
                        # Should have removed some stuttering
                        total_removed = result['repetitions_removed'] + result['pauses_removed']
                        if total_removed == 0 and reduction > 5:
                            issues.append("Duration reduced but no stuttering detected")
                        
                        if issues:
                            results['issues'].append({
                                'type': 'WARNING',
                                'component': f'Real-world Test {i+1}',
                                'issue': '; '.join(issues),
                                'file': test_file
                            })
                            print(f"      ⚠️  Issues: {'; '.join(issues)}")
                        else:
                            print(f"      ✅ Scenario {i+1} passed")
                        
                        # Clean up
                        os.remove(output_file)
                        
                    else:
                        results['issues'].append({
                            'type': 'ERROR',
                            'component': f'Real-world Test {i+1}',
                            'issue': 'Output file not created',
                            'file': test_file
                        })
                    
                    results['scenarios_tested'] += 1
                    
                except Exception as e:
                    results['issues'].append({
                        'type': 'ERROR',
                        'component': f'Real-world Test {i+1}',
                        'issue': f"Testing failed: {str(e)}",
                        'file': test_file,
                        'traceback': traceback.format_exc()
                    })
                    print(f"      ❌ Scenario {i+1} failed: {str(e)}")
            
        except Exception as e:
            results['issues'].append({
                'type': 'ERROR',
                'component': 'Real-world Testing',
                'issue': f"Real-world testing setup failed: {str(e)}",
                'traceback': traceback.format_exc()
            })
        
        print(f"  📊 Real-world testing: {results['scenarios_tested']} scenarios tested")
        return results
    
    def assess_critical_issues(self, analysis_results: Dict) -> List[Dict]:
        """Assess and prioritize critical issues"""
        all_issues = (analysis_results['functional_issues'] + 
                     analysis_results['performance_issues'])
        
        critical_issues = []
        
        for issue in all_issues:
            # Mark as critical based on type and impact
            if issue.get('type') == 'CRITICAL':
                critical_issues.append(issue)
            elif issue.get('type') == 'ERROR' and 'cannot function' in issue.get('issue', ''):
                critical_issues.append(issue)
            elif 'component' in issue and issue['component'] in ['Config', 'Preprocessing', 'Segmentation']:
                if issue.get('type') in ['ERROR', 'CRITICAL']:
                    critical_issues.append(issue)
        
        print(f"  🚨 Found {len(critical_issues)} critical issues")
        for i, issue in enumerate(critical_issues, 1):
            print(f"    {i}. {issue.get('component', 'Unknown')}: {issue.get('issue', 'No description')}")
        
        return critical_issues
    
    def generate_comprehensive_fixes(self, analysis_results: Dict) -> List[Dict]:
        """Generate comprehensive fixes for all identified issues"""
        fixes = []
        
        print("  🔧 Generating fixes...")
        
        # Fix for multiple pipeline confusion
        pipeline_files = ['pipeline.py', 'fixed_pipeline.py', 'balanced_pipeline.py', 'conservative_pipeline.py']
        existing_pipelines = [f for f in pipeline_files if os.path.exists(f)]
        
        if len(existing_pipelines) > 1:
            fixes.append({
                'priority': 'HIGH',
                'type': 'ARCHITECTURE',
                'issue': 'Multiple pipeline implementations causing confusion',
                'fix': 'Consolidate to single working pipeline (conservative_pipeline.py)',
                'action': 'Rename conservative_pipeline.py to pipeline.py and backup original'
            })
        
        # Fix component issues
        for issue in analysis_results['functional_issues']:
            if issue.get('type') == 'ERROR':
                component = issue.get('component', 'Unknown')
                fixes.append({
                    'priority': 'HIGH',
                    'type': 'COMPONENT',
                    'issue': f"{component} component failing",
                    'fix': f"Fix {component} implementation",
                    'action': f"Debug and repair {component}.py"
                })
        
        # Fix performance issues
        for issue in analysis_results['performance_issues']:
            fixes.append({
                'priority': 'MEDIUM',
                'type': 'PERFORMANCE',
                'issue': issue.get('issue', 'Performance problem'),
                'fix': 'Optimize processing speed',
                'action': 'Implement vectorization and reduce computational complexity'
            })
        
        # Fix real-world issues
        for issue in analysis_results['functional_issues']:
            if issue.get('component', '').startswith('Real-world Test'):
                fixes.append({
                    'priority': 'MEDIUM',
                    'type': 'ALGORITHM',
                    'issue': issue.get('issue', 'Real-world performance issue'),
                    'fix': 'Adjust algorithm parameters for better real-world performance',
                    'action': 'Fine-tune thresholds and detection logic'
                })
        
        print(f"  📋 Generated {len(fixes)} fixes")
        for i, fix in enumerate(fixes, 1):
            print(f"    {i}. [{fix['priority']}] {fix['issue']}")
        
        return fixes
    
    def print_final_summary(self, analysis_results: Dict):
        """Print comprehensive final summary"""
        print(f"\n🎯 PROFESSIONAL ANALYSIS SUMMARY")
        print("=" * 60)
        
        print(f"System Status: {analysis_results['system_status']}")
        print(f"Critical Issues: {len(analysis_results['critical_issues'])}")
        print(f"Functional Issues: {len(analysis_results['functional_issues'])}")
        print(f"Performance Issues: {len(analysis_results['performance_issues'])}")
        print(f"Fixes Required: {len(analysis_results['fixes_needed'])}")
        
        if analysis_results['critical_issues']:
            print(f"\n🚨 CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION:")
            for issue in analysis_results['critical_issues']:
                print(f"   • {issue.get('component', 'Unknown')}: {issue.get('issue', 'No description')}")
        
        if analysis_results['fixes_needed']:
            print(f"\n🔧 RECOMMENDED FIXES:")
            high_priority = [f for f in analysis_results['fixes_needed'] if f['priority'] == 'HIGH']
            medium_priority = [f for f in analysis_results['fixes_needed'] if f['priority'] == 'MEDIUM']
            
            if high_priority:
                print(f"   HIGH PRIORITY:")
                for fix in high_priority:
                    print(f"     • {fix['issue']}")
            
            if medium_priority:
                print(f"   MEDIUM PRIORITY:")
                for fix in medium_priority:
                    print(f"     • {fix['issue']}")
        
        # Overall assessment
        total_issues = len(analysis_results['critical_issues']) + len(analysis_results['functional_issues'])
        if total_issues == 0:
            print(f"\n✅ SYSTEM IS HEALTHY - No issues found!")
        elif len(analysis_results['critical_issues']) == 0:
            print(f"\n⚠️  SYSTEM NEEDS MINOR FIXES - Functional but can be improved")
        else:
            print(f"\n❌ SYSTEM NEEDS MAJOR FIXES - Critical issues must be resolved")
    
    def apply_critical_fixes(self, analysis_results: Dict) -> bool:
        """Apply critical fixes automatically"""
        print(f"\n🔧 APPLYING CRITICAL FIXES")
        print("=" * 50)
        
        fixes_applied = 0
        
        for fix in analysis_results['fixes_needed']:
            if fix['priority'] == 'HIGH' and fix['type'] == 'ARCHITECTURE':
                if 'multiple pipeline implementations' in fix['issue']:
                    print(f"  🔄 Fixing multiple pipeline issue...")
                    
                    # Backup original pipeline
                    if os.path.exists('pipeline.py'):
                        os.rename('pipeline.py', 'pipeline_original_backup.py')
                        print(f"    ✅ Backed up original pipeline.py")
                    
                    # Use conservative pipeline as main
                    if os.path.exists('conservative_pipeline.py'):
                        os.rename('conservative_pipeline.py', 'pipeline.py')
                        print(f"    ✅ Set conservative pipeline as main pipeline")
                        fixes_applied += 1
                    
                    break
        
        print(f"\n📊 Applied {fixes_applied} critical fixes")
        return fixes_applied > 0


def main():
    """Run professional debugging and analysis"""
    debugger = ProfessionalDebugger()
    
    # Run comprehensive analysis
    results = debugger.run_comprehensive_analysis()
    
    # Apply critical fixes if needed
    if results['critical_issues']:
        print(f"\n🚨 CRITICAL ISSUES DETECTED - APPLYING FIXES...")
        fixes_applied = debugger.apply_critical_fixes(results)
        
        if fixes_applied:
            print(f"✅ Critical fixes applied. Please re-run the analysis.")
        else:
            print(f"⚠️  Some fixes require manual intervention.")
    else:
        print(f"\n✅ No critical issues found. System is ready for use!")
    
    return results


if __name__ == "__main__":
    results = main()
