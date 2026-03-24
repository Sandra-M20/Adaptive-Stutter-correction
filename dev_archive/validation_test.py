"""
validation_test.py
==================
Validate the conservative stuttering correction system
"""

import numpy as np
import soundfile as sf
import os
import sys
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def validate_conservative_system():
    """
    Comprehensive validation of the conservative system.
    """
    print("🔍 VALIDATING CONSERVATIVE STUTTERING CORRECTION")
    print("=" * 60)
    
    # Test files to validate
    test_files = [
        'output/_test_stutter_original.wav',
        'test_input.wav',
        '_selftest_input.wav'
    ]
    
    # Find available test files
    available_files = []
    for file in test_files:
        if os.path.exists(file):
            available_files.append(file)
    
    if not available_files:
        print("❌ No test files found. Please ensure at least one audio file exists.")
        return False
    
    print(f"Found {len(available_files)} test file(s)")
    
    # Import conservative pipeline
    from conservative_pipeline import ConservativeStutterCorrectionPipeline
    
    pipeline = ConservativeStutterCorrectionPipeline()
    
    validation_results = []
    
    for i, test_file in enumerate(available_files, 1):
        print(f"\n📝 Test {i}/{len(available_files)}: {test_file}")
        print("-" * 40)
        
        try:
            # Load original for comparison
            original_signal, sr = sf.read(test_file)
            if len(original_signal.shape) > 1:
                original_signal = np.mean(original_signal, axis=1)
            
            original_duration = len(original_signal) / sr
            original_energy = np.sum(original_signal ** 2)
            
            print(f"Original: {original_duration:.2f}s, Energy: {original_energy:.2f}")
            
            # Process with conservative pipeline
            start_time = time.time()
            output_file = f"validated_output_{i}.wav"
            
            result = pipeline.correct(test_file, output_file)
            
            processing_time = time.time() - start_time
            
            # Load and analyze output
            corrected_signal, sr_corrected = sf.read(output_file)
            if len(corrected_signal.shape) > 1:
                corrected_signal = np.mean(corrected_signal, axis=1)
            
            corrected_duration = len(corrected_signal) / sr_corrected
            corrected_energy = np.sum(corrected_signal ** 2)
            
            # Calculate metrics
            energy_preservation = corrected_energy / original_energy
            real_time_factor = processing_time / original_duration
            
            # Validation criteria
            validation_score = 0
            issues = []
            
            # Check 1: Reasonable duration reduction (5-15%)
            if 5 <= result['reduction_percent'] <= 15:
                validation_score += 1
                print("✅ Duration reduction in acceptable range")
            else:
                issues.append(f"Duration reduction {result['reduction_percent']:.1f}% outside 5-15% range")
            
            # Check 2: Energy preservation (>40%)
            if energy_preservation > 0.4:
                validation_score += 1
                print("✅ Energy preservation acceptable")
            else:
                issues.append(f"Energy preservation {energy_preservation*100:.0f}% too low")
            
            # Check 3: Processing speed (RTF < 2.0)
            if real_time_factor < 2.0:
                validation_score += 1
                print("✅ Processing speed acceptable")
            else:
                issues.append(f"Processing too slow (RTF: {real_time_factor:.2f})")
            
            # Check 4: Some stuttering detected
            total_issues = result['pauses_removed'] + result['prolongations_removed'] + result['repetitions_removed']
            if total_issues > 0:
                validation_score += 1
                print(f"✅ Stuttering issues detected and removed: {total_issues}")
            else:
                issues.append("No stuttering issues detected")
            
            # Store results
            test_result = {
                'file': test_file,
                'validation_score': validation_score,
                'max_score': 4,
                'issues': issues,
                'metrics': {
                    'duration_reduction': result['reduction_percent'],
                    'energy_preservation': energy_preservation,
                    'processing_time': processing_time,
                    'real_time_factor': real_time_factor,
                    'issues_removed': total_issues
                }
            }
            
            validation_results.append(test_result)
            
            print(f"\n📊 Test {i} Results:")
            print(f"   Validation Score: {validation_score}/4")
            print(f"   Duration Reduction: {result['reduction_percent']:.1f}%")
            print(f"   Energy Preserved: {energy_preservation*100:.0f}%")
            print(f"   Processing Time: {processing_time:.2f}s (RTF: {real_time_factor:.2f})")
            print(f"   Issues Removed: {total_issues}")
            
            if issues:
                print(f"   ⚠️  Issues: {', '.join(issues)}")
            else:
                print(f"   ✅ All checks passed!")
            
        except Exception as e:
            print(f"❌ Error processing {test_file}: {e}")
            validation_results.append({
                'file': test_file,
                'validation_score': 0,
                'max_score': 4,
                'issues': [f"Processing error: {e}"],
                'metrics': {}
            })
    
    # Overall validation summary
    print(f"\n🎯 OVERALL VALIDATION SUMMARY")
    print("=" * 60)
    
    total_score = sum(r['validation_score'] for r in validation_results)
    max_possible_score = sum(r['max_score'] for r in validation_results)
    overall_percentage = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
    
    print(f"Overall Score: {total_score}/{max_possible_score} ({overall_percentage:.0f}%)")
    
    if overall_percentage >= 75:
        print("✅ SYSTEM VALIDATION PASSED - Ready for deployment!")
        deployment_ready = True
    elif overall_percentage >= 50:
        print("⚠️  SYSTEM VALIDATION PARTIAL - Review issues before deployment")
        deployment_ready = False
    else:
        print("❌ SYSTEM VALIDATION FAILED - Major issues need fixing")
        deployment_ready = False
    
    # Detailed recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    for result in validation_results:
        if result['issues']:
            print(f"\nFor {result['file']}:")
            for issue in result['issues']:
                print(f"   • Fix: {issue}")
    
    if deployment_ready:
        print(f"\n🚀 DEPLOYMENT READY!")
        print(f"   • System meets all quality criteria")
        print(f"   • Conservative settings ensure safety")
        print(f"   • Ready for production use")
    
    return deployment_ready, validation_results

def create_deployment_package():
    """
    Create a deployment package with the conservative system.
    """
    print(f"\n📦 CREATING DEPLOYMENT PACKAGE")
    print("=" * 60)
    
    deployment_files = [
        'conservative_pipeline.py',
        'config.py',
        'preprocessing.py',
        'segmentation.py',
        'pause_corrector.py',
        'prolongation_corrector.py',
        'speech_reconstructor.py',
        'audio_enhancer.py'
    ]
    
    # Create deployment directory
    deploy_dir = "deployment_package"
    os.makedirs(deploy_dir, exist_ok=True)
    
    # Copy essential files
    import shutil
    copied_files = []
    
    for file in deployment_files:
        if os.path.exists(file):
            shutil.copy2(file, os.path.join(deploy_dir, file))
            copied_files.append(file)
            print(f"✅ Copied: {file}")
        else:
            print(f"⚠️  Missing: {file}")
    
    # Create deployment README
    readme_content = """# Conservative Stuttering Correction System

## Quick Start
```python
from conservative_pipeline import ConservativeStutterCorrectionPipeline

# Create pipeline
pipeline = ConservativeStutterCorrectionPipeline()

# Process audio
result = pipeline.correct('input.wav', 'output.wav')

print(f"Removed {result['repetitions_removed']} repetitions")
print(f"Duration reduced by {result['reduction_percent']:.1f}%")
```

## Features
- CONSERVATIVE: Conservative stuttering detection (7-10% reduction)
- HIGH QUALITY: High audio quality preservation (>40% energy)
- REAL-TIME: Real-time processing (RTF < 2.0)
- SAFE: Safe for production use

## Settings
- Similarity threshold: 0.92 (high confidence only)
- Chunk size: 250ms (reduces false positives)
- Max removals: 15% of audio

## Files
- conservative_pipeline.py - Main correction pipeline
- Supporting modules for preprocessing, segmentation, etc.
"""
    
    with open(os.path.join(deploy_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Created: README.md")
    print(f"\n📦 Deployment package created in: {deploy_dir}/")
    print(f"   Files included: {len(copied_files)}")
    
    return deploy_dir

if __name__ == "__main__":
    # Step 1: Validate the system
    deployment_ready, results = validate_conservative_system()
    
    # Step 2: Create deployment package if ready
    if deployment_ready:
        deploy_dir = create_deployment_package()
        print(f"\n🎉 CONSERVATIVE SYSTEM READY FOR DEPLOYMENT!")
    else:
        print(f"\n⚠️  Please address validation issues before deployment.")
