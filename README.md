<div align="center">
<img src="./resources/logos/logo.jpeg" width="350">
</div>

&nbsp;
&nbsp;
# ml-accelerator
ml_accelerator is a propietary library designed to accelerate ML proyects.

&nbsp;
# Table of Contents

- [Installation](#installation)
- [Usage](#usage)

&nbsp;
# Installation

1. Install the AWS CLI v2 (if it's not already installed)
```bash
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg ./AWSCLIV2.pkg -target /
```
2. Set up the IAM credentials using aws configure:
```bash
aws configure
```
```
AWS Access Key ID: AWS_ACCESS_KEY_ID
AWS Secret Access Key: AWS_SECRET_ACCESS_KEY
Default region name: sa-east-1
Default output format: json
```
3. Clone the `ml-accelerator` CodeCommit repository:
```bash
git clone ...
```
4. Create & activate python virtual environment:
```bash
python -m venv .ml_accel_venv
source .ml_accel_venv/bin/activate
```
5. Install the ml_accelerator module in "editable" mode:
```bash
pip install -e .
```
  - *Note that this command will also install the dependencies, specified in `requirements.txt`.*
6. Install & run the [Docker Desktop](https://docs.docker.com/engine/install/) application (if it's not already installed). 