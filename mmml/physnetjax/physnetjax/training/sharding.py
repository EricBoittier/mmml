
    SHARDING = False
    if SHARDING:
        """
        Data sharding
        """
        num_devices = len(jax.local_devices())
        print(f"Running on {num_devices} devices: {jax.local_devices()}")
        devices = mesh_utils.create_device_mesh((num_devices,))

        # Data will be split along the batch axis
        data_mesh = Mesh(devices, axis_names=("batch",))  # naming axes of the mesh
        data_sharding = NamedSharding(
            data_mesh,
            P(
                "batch",
            ),
        )  # naming axes of the sharded partition

        # Display data sharding
        x = next(iter(train_data))
        for y in train_data:
            train_data[y] = jax.device_put(train_data[y],
                                           data_sharding)
            print(f"Data sharding for {y}")
            if len(train_data[y].shape) < 2:
                jax.debug.visualize_array_sharding(train_data[y])
        ####################################################################