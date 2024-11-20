import os

configfile: "config.yaml"

PGEN_EXT = ["pgen", "pvar", "psam"]
BED_EXT = ["bed", "bim", "fam"]
RAW_EXT = ["raw", "log"]
SPLIT_OUTPUT = config.get("output", {}).get("split", False)
IN_FILE = os.path.basename(config.get("input", {}).get("data", ""))

rule all:
    input:
        f"{config['output']['dir']}/train/{IN_FILE}_shuffled_train.hdf5",
        f"{config['output']['dir']}/test/{IN_FILE}_shuffled_test.hdf5",
        expand(
            "{outdir}/{split}/{infile}_shuffled_{split}.{ext}",
            outdir=config['output']['dir'],
            split=["train", "test"],
            infile=IN_FILE,
            ext=PGEN_EXT if config['input']['file_type'] == 'pfile' else BED_EXT
        )

rule download_plink:
    output:
        "bin/plink2"
    params:
        url=config['input']['plink_url']
    shell:
        """
        mkdir -p bin
        wget -O bin/plink2.zip {params.url}
        unzip bin/plink2.zip -d bin/
        rm -f bin/plink2.zip
        chmod +x bin/plink2
        """

rule convert_plink_to_raw:
    input:
        "bin/plink2",
        expand("{inpath}.{ext}", inpath=config["input"]["data"], ext=PGEN_EXT if config['input']['file_type'] == 'pfile' else BED_EXT)
    params:
        infile=f"{config['input']['data']}",
        allelefile=f"{config['input']['allele_file']}",
        outfile=f"{config['output']['dir']}/tmp/{IN_FILE}",
        tool=config['input']['tool'],
        file_type=config['input']['file_type']
    output:
        expand("{outdir}/tmp/{inpath}.{ext}",
               outdir=config["output"]["dir"],
               inpath=IN_FILE,
               ext=RAW_EXT)
    threads:
        config["run"]["threads"]
    shell:
        """
        {params.tool} \
            --{params.file_type} {params.infile} \
            --recode A \
            --recode-allele {params.allelefile} \
            --out {params.outfile}
        """

rule shuffle_raw_file:
    input:
        expand("{outdir}/tmp/{inpath}.{ext}",
               outdir=config["output"]["dir"],
               inpath=IN_FILE,
               ext=RAW_EXT)
    params:
        infile=f"{config['output']['dir']}/tmp/{IN_FILE}.raw"
    output:
        f"{config['output']['dir']}/tmp/{IN_FILE}_shuffled.raw"
    threads:
        config["run"]["threads"]
    shell:
        """
        awk '(NR == 1) {{print $0}}' {params.infile} > {output}
        awk '(NR > 1) {{print $0}}' {params.infile} | shuf >> {output}
        """

rule convert_raw_to_hdf5:
    input:
        f"{config['output']['dir']}/tmp/{IN_FILE}_shuffled.raw"
    params:
        read_chunks=f"{config['input']['chunk_size']}",
        dask_chunks=f"{config['output']['chunk_size']}",
        dask_dtype=f"{config['output']['dtype']}",
        tmp_file=f"{config['output']['dir']}/tmp/{IN_FILE}_shuffled.hdf5",
        out_dir=config['output']['dir']
    output:
        f"{config['output']['dir']}/{IN_FILE}_shuffled.hdf5"
    threads:
        config["run"]["threads"]
    shell:
        """
        wc_raw=$(wc -l < {input})
        nrow_raw=$((wc_raw - 1))
        
        python3 scripts/raw_to_hdf5.py \
            --raw {input} \
            --nrows $nrow_raw \
            --dask_chunks {params.dask_chunks} \
            --read_raw_chunk_size {params.read_chunks} \
            --dtype {params.dask_dtype}
        
        mv {params.tmp_file} {params.out_dir}
        """

rule generate_train_test_split:
    input:
        f"{config['output']['dir']}/{IN_FILE}_shuffled.hdf5",
        f"{config['input']['covar_file']}"
    params:
        infile=config['input']['covar_file'],
        outdir=config['output']['dir'],
        p_split=config['output']['p_split'],
        seed=config['run']['seed']
    output:
        f"{config['output']['dir']}/train/train_ids.txt",
        f"{config['output']['dir']}/test/test_ids.txt",
        f"{config['output']['dir']}/train/train_ids_plinkformat.txt",
        f"{config['output']['dir']}/test/test_ids_plinkformat.txt"
    threads:
        config["run"]["threads"]
    run:
        if not SPLIT_OUTPUT:
            print("Warning: skipping train/test split as split is false or not set in config file")
        else:
            shell(
                """
                python3 scripts/split_ids.py \
                    --file {params.infile} \
                    --proportion {params.p_split} \
                    --out_dir {params.outdir} \
                    --seed {params.seed}
                """
            )

rule split_hdf5_file:
    input:
        f"{config['output']['dir']}/{IN_FILE}_shuffled.hdf5",
        f"{config['output']['dir']}/train/train_ids.txt",
        f"{config['output']['dir']}/test/test_ids.txt"
    params:
        infile=f"{config['output']['dir']}/{IN_FILE}_shuffled.hdf5",
        train_outfile=f"{config['output']['dir']}/train/{IN_FILE}_shuffled_train.hdf5",
        test_outfile=f"{config['output']['dir']}/test/{IN_FILE}_shuffled_test.hdf5",
        train_ids=f"{config['output']['dir']}/train/train_ids.txt",
        test_ids=f"{config['output']['dir']}/test/test_ids.txt",
        row_chunks=config['output']['chunk_size']
    output:
        f"{config['output']['dir']}/train/{IN_FILE}_shuffled_train.hdf5",
        f"{config['output']['dir']}/test/{IN_FILE}_shuffled_test.hdf5"
    threads:
        config["run"]["threads"]
    run:
        if not SPLIT_OUTPUT:
            print("Warning: skipping hdf5 file split as split is false or not set in config file")
        else:
            shell(
                """
                # subset train IDs and write to separate hdf5 file
                python3 scripts/split_hdf5.py \
                    --in_path {params.infile} \
                    --out_path {params.train_outfile} \
                    --ids {params.train_ids} \
                    --row_chunks {params.row_chunks} \
                    --xkey x \
                    --ykey y
                
                # subset test IDs and write to separate hdf5 file
                python3 scripts/split_hdf5.py \
                    --in_path {params.infile} \
                    --out_path {params.test_outfile} \
                    --ids {params.test_ids} \
                    --row_chunks {params.row_chunks} \
                    --xkey x \
                    --ykey y
                """
            )

rule split_plink_file:
    input:
        "bin/plink2",
        expand("{inpath}.{ext}", inpath=config["input"]["data"], ext=PGEN_EXT if config['input']['file_type'] == 'pfile' else BED_EXT),
        f"{config['output']['dir']}/train/train_ids_plinkformat.txt",
        f"{config['output']['dir']}/test/test_ids_plinkformat.txt"
    params:
        infile=f"{config['input']['data']}",
        tool=config['input']['tool'],
        file_type=config['input']['file_type'],
        write_type='make-pgen' if config['output']['file_type'] == 'pfile' else 'make-bed',
        train_ids=f"{config['output']['dir']}/train/train_ids_plinkformat.txt",
        test_ids=f"{config['output']['dir']}/test/test_ids_plinkformat.txt",
        train_outfile=f"{config['output']['dir']}/train/{IN_FILE}_shuffled_train",
        test_outfile=f"{config['output']['dir']}/test/{IN_FILE}_shuffled_test"
    output:
        expand(
            "{outdir}/{split}/{infile}_shuffled_{split}.{ext}",
            outdir=config['output']['dir'],
            split=["train", "test"],
            infile=IN_FILE,
            ext=PGEN_EXT if config['input']['file_type'] == 'pfile' else BED_EXT
        )
    threads:
        config["run"]["threads"]
    run:
        if not SPLIT_OUTPUT:
            print("Warning: skipping hdf5 file split as split is false or not set in config file")
        else:
            shell(
                """
                # split train data for plink file
                {params.tool} \
                    --{params.file_type} {params.infile} \
                    --keep {params.train_ids} \
                    --out {params.train_outfile} \
                    --{params.write_type}
                
                # split test data for plink file
                {params.tool} \
                    --{params.file_type} {params.infile} \
                    --keep {params.test_ids} \
                    --out {params.test_outfile} \
                    --{params.write_type}
                """
            )

# add covariate adjustment - optional, non-urgent
# make sure download for plink runs on head node but all else in slurm jobs
# needs python env attached to the python stuff
# - 