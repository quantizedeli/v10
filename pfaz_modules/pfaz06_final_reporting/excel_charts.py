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

    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir) if output_dir else Path('reports')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Excel Chart Generator başlatıldı")

    def generate_all_charts(self, all_results):
        """Tum sonuclar icin chart'lar olustur ve kaydet"""
        import pandas as pd
        charts = []
        try:
            # AI models chart — yeni yapi: ai_rows listesi
            ai_rows = all_results.get('ai_rows', [])
            if not ai_rows:
                # Geriye donuk uyumluluk: eski ai_models dict yapisi
                for model_name, row in all_results.get('ai_models', {}).items():
                    if isinstance(row, dict) and 'Val_R2' in row:
                        ai_rows.append(row)

            if ai_rows:
                df = pd.DataFrame(ai_rows)
                # Remove diverged/extreme R2 rows before charting
                for _col in ['Val_R2', 'Test_R2', 'Train_R2']:
                    if _col in df.columns:
                        df = df[pd.to_numeric(df[_col], errors='coerce').fillna(0) >= -10]
                chart_file = self.output_dir / 'ai_models_chart.xlsx'
                with pd.ExcelWriter(chart_file, engine='openpyxl') as writer:
                    # Buyuk veri: sadece ozet (en iyi 2000 kayit Val_R2'ye gore)
                    top_df = df.nlargest(2000, 'Val_R2') if 'Val_R2' in df.columns else df.head(2000)
                    top_df.to_excel(writer, sheet_name='AI_Top2000', index=False)
                    # Model-tipi bazli ozetler
                    if 'Model_Type' in df.columns:
                        for mtype in df['Model_Type'].unique():
                            sub = df[df['Model_Type'] == mtype].sort_values('Val_R2', ascending=False)
                            sub.head(200).to_excel(writer, sheet_name=f'{mtype}_Top200'[:31], index=False)
                    # Dataset ozet
                    if 'Dataset' in df.columns and 'Val_R2' in df.columns:
                        ds_summary = df.groupby('Dataset').agg(
                            Best_Val_R2=('Val_R2','max'),
                            Mean_Val_R2=('Val_R2','mean'),
                            N_Configs=('Config_ID','count')
                        ).reset_index().sort_values('Best_Val_R2', ascending=False)
                        ds_summary.to_excel(writer, sheet_name='Dataset_Summary', index=False)
                charts.append(str(chart_file))
                logger.info(f"[OK] AI models chart: {chart_file}")

            # ANFIS models chart — yeni yapi: anfis_results listesi
            anfis_rows = all_results.get('anfis_results', [])
            if anfis_rows:
                df = pd.DataFrame(anfis_rows)
                df_disp = df.drop(columns=['Workspace_MAT', 'FIS_MAT'], errors='ignore')
                chart_file = self.output_dir / 'anfis_models_chart.xlsx'
                with pd.ExcelWriter(chart_file, engine='openpyxl') as writer:
                    df_disp.to_excel(writer, sheet_name='ANFIS_Models', index=False)
                charts.append(str(chart_file))
                logger.info(f"[OK] ANFIS models chart: {chart_file}")

        except Exception as e:
            logger.warning(f"[WARNING] Chart generation partial failure: {e}")

        return charts
    
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
            
        logger.info(f"[OK] Excel with charts: {output_file}")
    
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
        
        logger.info(f"[OK] Performance chart added to {sheet_name}")


if __name__ == "__main__":
    print("[OK] Excel Charts modülü hazır - visualization/excel_charts.py")