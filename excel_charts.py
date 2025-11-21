"""
Excel Charts Module
Excel içinde embedded grafikler

10. modül - visualization/excel_charts.py
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExcelChartGenerator:
    """Excel içinde grafikler oluştur"""
    
    def __init__(self):
        logger.info("Excel Chart Generator başlatıldı")
    
    def create_excel_with_charts(self, data_df, output_file, chart_configs):
        """
        Excel dosyasında grafikler oluştur
        
        Args:
            data_df: Data DataFrame
            output_file: Output Excel file
            chart_configs: List of chart configurations
        """
        
        output_file = Path(output_file)
        
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Write data
            data_df.to_excel(writer, sheet_name='Data', index=False)
            worksheet = writer.sheets['Data']
            
            # Add charts
            for i, config in enumerate(chart_configs):
                chart = self._create_chart(workbook, config)
                worksheet.insert_chart(f'K{i*15 + 2}', chart)
            
        logger.info(f"✓ Excel with charts: {output_file}")
    
    def _create_chart(self, workbook, config):
        """Create single chart"""
        
        chart_type = config.get('type', 'column')
        chart = workbook.add_chart({'type': chart_type})
        
        # Add series
        for series_config in config.get('series', []):
            chart.add_series(series_config)
        
        # Chart formatting
        chart.set_title({'name': config.get('title', '')})
        chart.set_x_axis({'name': config.get('x_label', '')})
        chart.set_y_axis({'name': config.get('y_label', '')})
        chart.set_style(config.get('style', 10))
        
        return chart
    
    def add_performance_chart(self, writer, data_df, sheet_name='Dashboard'):
        """Add model performance comparison chart"""
        workbook = writer.book
        worksheet = workbook.add_worksheet(sheet_name)
        
        # Write summary data
        summary = data_df.groupby('Model')['R2_test'].mean().reset_index()
        summary.to_excel(writer, sheet_name=sheet_name, startrow=1, startcol=0, index=False)
        
        # Create chart
        chart = workbook.add_chart({'type': 'column'})
        
        chart.add_series({
            'name': 'Average R²',
            'categories': f'={sheet_name}!$A$3:$A${len(summary)+2}',
            'values': f'={sheet_name}!$B$3:$B${len(summary)+2}',
            'fill': {'color': '#4472C4'},
            'border': {'color': 'black'}
        })
        
        chart.set_title({'name': 'Model Performance Comparison'})
        chart.set_x_axis({'name': 'Model'})
        chart.set_y_axis({'name': 'Average R² Score'})
        chart.set_style(10)
        
        worksheet.insert_chart('D2', chart, {'x_scale': 1.5, 'y_scale': 1.5})
        
        logger.info(f"✓ Performance chart added to {sheet_name}")


if __name__ == "__main__":
    print("✓ Excel Charts modülü hazır - visualization/excel_charts.py")