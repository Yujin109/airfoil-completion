.PHONY: build-xfoil
build-xfoil: ## DCMAKE_Fortran_COMPILER={gfortranのpath}(which gfortranで確認)
	cd ./xfoil-python
	python old_setup.py build_ext -- -DCMAKE_Fortran_COMPILER=/opt/homebrew/bin/gfortran
	cd ..