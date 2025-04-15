.PHONY: setup-xfoil
setup-xfoil:
	sudo apt update
	sudo apt install -y gfortran
	sudo apt install -y cmake
	git clone https://github.com/Yujin109/xfoil-python.git || true
	cd xfoil-python && python old_setup.py build_ext
	pip install --no-cache-dir --force-reinstall ./xfoil-python
	cp xfoil-python/build/lib.linux-x86_64-cpython-310/xfoil/libxfoil.so /opt/conda/lib/python3.10/site-packages/xfoil/libxfoil.so
	/opt/conda/envs/pytorch/bin/pip install --no-cache-dir --force-reinstall ./xfoil-python
	cp xfoil-python/build/lib.linux-x86_64-cpython-310/xfoil/libxfoil.so /opt/conda/envs/pytorch/lib/python3.10/site-packages/xfoil/libxfoil.so