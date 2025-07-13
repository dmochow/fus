# Initialize and run
from minimal_preprocessing_pipeline import MinimalPreprocessor

def run():
    preprocessor = MinimalPreprocessor("/Users/jacekdmochowski/PROJECTS/fus/data/resampled_bold_flywheel/")

    # Process with minimal approach (recommended first)
    qc_minimal = preprocessor.process_all_sessions(processing_level='minimal')

    # Compare with conservative approach if needed
    qc_conservative = preprocessor.process_all_sessions(processing_level='conservative')

    return qc_minimal, qc_conservative

if __name__ == "__main__":
    qc_minimal, qc_conservative = run()
    print("Minimal QC results:", qc_minimal)
    print("Conservative QC results:", qc_conservative)


