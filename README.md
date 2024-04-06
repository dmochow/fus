## Dmochowski Lab - FUS-fMRI Project
The project combines low intensity transcranial focused ultrasound (tFUS) stimulation with 
functional Magnetic Resonance Imaging (fMRI) in order to understand the effects of tFUS on
human brain activity. 

## Data
The data is stored on the cloud via the Flywheel platform. The login page is [here](https://cuny-mri.flywheel.io/).

One of our tasks will be to use the Flywheel Python SDK to access the data. The SDK of the latest version is available [here](https://flywheel-io.gitlab.io/product/backend/sdk/tags/18.1.1/python/index.html).

## Code
Our code will be stored in the [code](./code) subfolder.

## Relevant software packages
- [fmriprep](https://fmriprep.org/en/stable/) Used for preprocessing fMRI data.
- [templateflow](https://www.templateflow.org/usage/client/) Used for downloading template images (atlases).
- [Nilearn](https://nilearn.github.io/) Used for analyzing fMRI data.