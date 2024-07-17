set -x

curl https://openmath-test-predictions.s3.amazonaws.com/openmath-test-predictions.zip --output ./examples/data/openmath-test-predictions.zip 

mkdir -p ./examples/data/base_model_generation/

unzip -j ./examples/data/openmath-test-predictions.zip "openmath-test-predictions/codellama-7b/algebra222/*" -d ./examples/data/base_model_generation/algebra222
unzip -j ./examples/data/openmath-test-predictions.zip "openmath-test-predictions/codellama-7b/asdiv/*" -d ./examples/data/base_model_generation/asdiv
unzip -j ./examples/data/openmath-test-predictions.zip "openmath-test-predictions/codellama-7b/gsm-hard/*" -d ./examples/data/base_model_generation/gsm-hard
unzip -j ./examples/data/openmath-test-predictions.zip "openmath-test-predictions/codellama-7b/gsm8k/*" -d ./examples/data/base_model_generation/gsm8k
unzip -j ./examples/data/openmath-test-predictions.zip "openmath-test-predictions/codellama-7b/math/*" -d ./examples/data/base_model_generation/math
unzip -j ./examples/data/openmath-test-predictions.zip "openmath-test-predictions/codellama-7b/mawps/*" -d ./examples/data/base_model_generation/mawps
unzip -j ./examples/data/openmath-test-predictions.zip "openmath-test-predictions/codellama-7b/svamp/*" -d ./examples/data/base_model_generation/svamp

unzip ./examples/data/preference_ranking_dataset.zip
rm ./examples/data/openmath-test-predictions.zip 

mkdir -p ./examples/data/preference_ranking_dataset/
unzip ./examples/data/preference_ranking_dataset.zip -d ./examples/data/