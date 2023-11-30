import click
import logging
import pandas as pd
from pathlib import Path

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Starting data processing')

    # Read the input CSV file
    logger.info(f'Reading input file: {input_filepath}')
    data = pd.read_csv(input_filepath)

    # Perform data processing (example: convert a column to uppercase)
    data['Column_to_convert'] = data['Column_to_convert'].str.upper()

    # Save the processed data to a new CSV file
    logger.info(f'Saving processed data to: {output_filepath}')
    data.to_csv(output_filepath, index=False)

    logger.info('Data processing completed')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
