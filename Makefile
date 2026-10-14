test:
	$(MAKE) -C tests

example:
	$(MAKE) -C examples

clean:
	$(MAKE) -C tests clean
	$(MAKE) -C examples clean

.PHONY: test example clean
