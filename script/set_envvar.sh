#!/bin/bash
##
## Copyright 2024 LY Corporation
##
## LY Corporation licenses this file to you under the Apache License,
## version 2.0 (the "License"); you may not use this file except in compliance
## with the License. You may obtain a copy of the License at:
##
##   https://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
## WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
## License for the specific language governing permissions and limitations
## under the License.
##

export TOKENIZERS_PARALLELISM="false"
export OPENAI_API_KEY="" # Required
export MLFLOW_TRACKING_URI="" # Optional
export HF_MLFLOW_LOG_ARTIFACTS="False" # change to True if you want to save artifact to AWS S3
export MLFLOW_S3_ENDPOINT_URL="" # Optional
export AWS_ACCESS_KEY_ID="" # Optional
export AWS_SECRET_ACCESS_KEY="" # Optional
