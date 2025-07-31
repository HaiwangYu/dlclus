import os
import sys
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button, Slider
from dlclus.prep.labeler import get_isnu_labels

class EnhancedEventDisplay:
    def __init__(self, rec_files, tru_files, initial_distance_cut=2.0, z_offset=0):
        """
        Interactive event display for visualization of reconstruction, truth, and labeled data.
        
        Parameters:
        -----------
        rec_files : list
            List of paths to reconstruction NPZ files
        tru_files : list
            List of paths to truth JSON files
        initial_distance_cut : float
            Initial value for distance cut in cm for matching points
        z_offset : float
            Initial value for z-axis offset in cm
        """
        self.rec_files = rec_files
        self.tru_files = tru_files
        self.current_event = 0
        self.num_events = len(rec_files)
        self.distance_cut = initial_distance_cut
        self.z_offset = z_offset
        self.view_mode = '2d_xz'  # Can be '3d', '2d_xy', '2d_xz', '2d_yz'
        
        # Load the first event
        self.load_current_event()
        
        # Setup the figure and plots
        self.fig = plt.figure(figsize=(20, 10))  # Increased size but still under 1920x1080
        self.setup_plot()
    
    def load_current_event(self):
        """Load the current event's reconstruction and truth data"""
        rec_file = self.rec_files[self.current_event]
        tru_file = self.tru_files[self.current_event]
        
        print(f"Loading event {self.current_event + 1}/{self.num_events}")
        print(f"Rec file: {rec_file}")
        print(f"Truth file: {tru_file}")
        
        # Load data files
        self.rec_data = load_rec_file(rec_file)
        self.tru_data = load_tru_file(tru_file)
        self.rec_file = rec_file
        self.tru_file = tru_file
        
        # Extract points from rec_data
        self.points = self.rec_data['points']
        print(f"self.points[0:10,:]: {self.points[0:10,:]}")
        print(f"self.rec_data['ppedges']: {self.rec_data['ppedges']}")
        print(f"self.rec_data['blobs']: {self.rec_data['blobs']}")
        self.x = self.points[:, 0]
        self.y = self.points[:, 1]
        self.z = self.points[:, 2] + self.z_offset*10  # mm
        self.point_color = self.points[:, 5] # 3: charge, 4: blob_idx; 5: cluster_idx
        
        # Apply initial labeling
        self.update_labels()
        
        # Extract truth points and labels for easier access
        self.extract_truth_data()
    
    def next_event(self, event=None):
        """Load the next event"""
        if self.current_event < self.num_events - 1:
            self.current_event += 1
            self.load_current_event()
            self.setup_plot()
    
    def prev_event(self, event=None):
        """Load the previous event"""
        if self.current_event > 0:
            self.current_event -= 1
            self.load_current_event()
            self.setup_plot()
        
    def extract_truth_data(self):
        """Extract truth data points and labels from the truth file"""
        self.tru_points = []
        self.tru_labels = []
        
        # Check if the truth data has the expected format
        if isinstance(self.tru_data, dict) and 'x' in self.tru_data and 'y' in self.tru_data and 'z' in self.tru_data:
            # Get coordinates from the flat structure
            x = np.array(self.tru_data.get('x', []))*10.
            y = np.array(self.tru_data.get('y', []))*10.
            z = np.array(self.tru_data.get('z', []))*10.
            q = np.array(self.tru_data.get('q', []))
            
            # Create points array if coordinates have consistent lengths
            if len(x) == len(y) == len(z):
                self.tru_points = np.column_stack((x, y, z))
                self.tru_labels = q if len(q) == len(x) else np.zeros(len(x))
                print(f"Extracted {len(self.tru_points)} truth points")
            else:
                print("Warning: Truth data coordinates have inconsistent lengths")
        else:
            print("Warning: Truth data does not have expected format")
    
    def update_labels(self):
        """Update labels based on current distance cut"""
        try:
            self.labels = get_isnu_labels(self.tru_file, self.rec_file, self.distance_cut, self.z_offset)
            print(f"Labels updated with distance cut = {self.distance_cut}")
            print(f"Label counts: {np.bincount(self.labels.astype(int) + 2)}")
        except Exception as e:
            print(f"Error updating labels: {str(e)}")
            self.labels = np.zeros(len(self.points))
    
    def setup_plot(self):
        """Set up the plot with panels and interactive controls"""
        # Clear previous figure content
        self.fig.clear()
        
        # Create subplot grid: 4 panels + space for controls
        if self.view_mode == '3d':
            self.ax_rec = self.fig.add_subplot(221, projection='3d')
            self.ax_tru = self.fig.add_subplot(222, projection='3d')
            self.ax_labeled = self.fig.add_subplot(223, projection='3d')
            self.ax_tagged = self.fig.add_subplot(224, projection='3d')
        else:
            self.ax_rec = self.fig.add_subplot(221)
            self.ax_tru = self.fig.add_subplot(222)
            self.ax_labeled = self.fig.add_subplot(223)
            self.ax_tagged = self.fig.add_subplot(224)
        
        # Add control buttons and slider
        self.add_controls()
        
        # Update the display
        self.update_display()
    
    def add_controls(self):
        """Add interactive controls to the figure"""
        # Define the right panel area for controls
        control_panel_width = 0.2
        control_panel_height = 0.9
        control_panel_bottom = 0.05
        control_panel_left = 0.78
        
        # Create a right panel for controls
        self.fig.subplots_adjust(right=control_panel_left - 0.01)
        
        # Add title for control panel
        plt.figtext(control_panel_left + control_panel_width/2, 0.95, "Controls", ha='center', fontsize=12)
        
        # Add buttons for view mode - vertically arranged on the right
        btn_height = 0.04
        btn_width = 0.1
        btn_spacing = 0.01
        
        # Starting positions
        y_pos = control_panel_bottom + control_panel_height - btn_height - 0.05
        
        # Add navigation buttons if multiple events
        if self.num_events > 1:
            # Add event navigation label
            plt.figtext(control_panel_left + control_panel_width/2, y_pos + btn_height, 
                       f"Event Navigation ({self.current_event + 1}/{self.num_events})", 
                       ha='center', fontsize=10)
            
            # Navigation buttons (horizontally arranged)
            ax_prev = plt.axes([control_panel_left, y_pos, btn_width, btn_height])
            ax_next = plt.axes([control_panel_left + btn_width + btn_spacing, y_pos, btn_width, btn_height])
            
            self.btn_prev = Button(ax_prev, '< Prev')
            self.btn_next = Button(ax_next, 'Next >')
            
            self.btn_prev.on_clicked(self.prev_event)
            self.btn_next.on_clicked(self.next_event)
            
            y_pos -= btn_height + 3*btn_spacing
        
        # View mode buttons (horizontally arranged)
        ax_3d = plt.axes([control_panel_left, y_pos, btn_width, btn_height])
        ax_xy = plt.axes([control_panel_left + btn_width + btn_spacing, y_pos, btn_width, btn_height])
        y_pos -= btn_height + 2*btn_spacing
        ax_xz = plt.axes([control_panel_left, y_pos, btn_width, btn_height])
        ax_yz = plt.axes([control_panel_left + btn_width + btn_spacing, y_pos, btn_width, btn_height])
        
        self.btn_3d = Button(ax_3d, '3D')
        self.btn_xy = Button(ax_xy, 'XY')
        self.btn_xz = Button(ax_xz, 'XZ')
        self.btn_yz = Button(ax_yz, 'YZ')
        
        self.btn_3d.on_clicked(self.view_3d)
        self.btn_xy.on_clicked(self.view_xy)
        self.btn_xz.on_clicked(self.view_xz)
        self.btn_yz.on_clicked(self.view_yz)
        
        # Add view mode label
        y_pos -= btn_spacing*3
        plt.figtext(control_panel_left + control_panel_width/2, y_pos, "View Mode", 
                   ha='center', fontsize=10)
        
        # Add sliders with labels above them
        slider_width = control_panel_width * 0.9
        slider_left = control_panel_left + (control_panel_width - slider_width)/2
        
        # Distance Cut slider
        y_pos -= btn_height + 2*btn_spacing
        # Add title above the slider
        plt.figtext(slider_left + slider_width/2, y_pos, "Distance Cut (cm)", 
                   ha='center', fontsize=10)
        y_pos -= btn_spacing * 2
        ax_slider = plt.axes([slider_left, y_pos, slider_width, btn_height])
        self.slider = Slider(
            ax_slider, '',  # Empty label since we're using figtext above
            0.0, 10.0,  # Range from 0 to 10 cm
            valinit=self.distance_cut,
            valstep=0.1
        )
        
        # Z Offset slider
        y_pos -= btn_height + 3*btn_spacing
        # Add title above the slider
        plt.figtext(slider_left + slider_width/2, y_pos, "Z Offset (cm)", 
                   ha='center', fontsize=10)
        y_pos -= btn_spacing * 2
        ax_z_slider = plt.axes([slider_left, y_pos, slider_width, btn_height])
        self.z_slider = Slider(
            ax_z_slider, '',  # Empty label since we're using figtext above
            -3.0, 3.0,  # Range from -3 to 3 cm
            valinit=self.z_offset,
            valstep=0.1
        )
        
        # Add button to update labels
        y_pos -= btn_height + 3*btn_spacing
        update_btn_width = slider_width
        update_btn_left = slider_left
        ax_update = plt.axes([update_btn_left, y_pos, update_btn_width, btn_height*1.2])
        self.btn_update = Button(ax_update, 'Update Labels')
        
        # Connect controls to functions
        self.btn_update.on_clicked(self.on_update_labels)
        self.slider.on_changed(self.on_slider_changed)
        self.z_slider.on_changed(self.on_z_offset_changed)
    
    def on_slider_changed(self, val):
        """Called when the distance slider value changes"""
        self.distance_cut = val
        self.update_status_text()
    
    def on_z_offset_changed(self, val):
        """Called when the z_offset slider value changes"""
        self.z_offset = val
        self.z = self.points[:, 2] + val*10  # Update z coordinates with new offset in mm
        self.update_display()
        self.update_status_text()
    
    def on_update_labels(self, event):
        """Called when the Update Labels button is clicked"""
        self.update_labels()
        self.update_display()
    
    def update_status_text(self):
        """Update the status text at the bottom of the figure"""
        # Clear previous text if it exists
        if hasattr(self, 'status_text'):
            try:
                self.status_text.remove()
            except:
                pass
        
        # Add new status text
        event_info = f"Event: {self.current_event + 1}/{self.num_events} | " if self.num_events > 1 else ""
        status = f"{event_info}Distance cut: {self.distance_cut:.1f} cm | Z offset: {self.z_offset:.1f} cm | View: {self.view_mode.upper().replace('2D_', '')}"
        self.status_text = plt.figtext(0.5, 0.01, status, ha='center', fontsize=10)
        plt.draw()
    
    def view_3d(self, event):
        """Switch to 3D view"""
        self.view_mode = '3d'
        self.setup_plot()
    
    def view_xy(self, event):
        """Switch to XY projection"""
        self.view_mode = '2d_xy'
        self.setup_plot()
    
    def view_xz(self, event):
        """Switch to XZ projection"""
        self.view_mode = '2d_xz'
        self.setup_plot()
    
    def view_yz(self, event):
        """Switch to YZ projection"""
        self.view_mode = '2d_yz'
        self.setup_plot()
    
    def update_display(self):
        """Update all four panels with current data and view"""
        # Clear axes
        self.ax_rec.clear()
        self.ax_tru.clear()
        self.ax_labeled.clear()
        self.ax_tagged.clear()
        
        # Plot differently based on view mode
        if self.view_mode == '3d':
            # Panel 1: Reconstruction data
            self.ax_rec.scatter(self.x, self.y, self.z, s=2, c=self.point_color, alpha=0.7)
            
            # Panel 2: Truth data
            tru_colors = np.array(['gray', 'green', 'blue', 'red'])[self.tru_labels.astype(int) + 2]
            if len(self.tru_points) > 0:
                # Use truth label for coloring if available
                if len(self.tru_labels) == len(self.tru_points):
                    self.ax_tru.scatter(self.tru_points[:, 0], self.tru_points[:, 1], 
                                        self.tru_points[:, 2], s=2, c=tru_colors,
                                        cmap='plasma', alpha=0.7)
                else:
                    self.ax_tru.scatter(self.tru_points[:, 0], self.tru_points[:, 1], 
                                        self.tru_points[:, 2], s=2, c='red', alpha=0.7)
            
            # Panel 3: Combined truth and reconstruction data
            # Plot reconstruction points in blue
            self.ax_labeled.scatter(self.x, self.y, self.z, s=2, c='blue', alpha=0.7, label='Reconstruction')
            # Plot truth points in red if available
            if len(self.tru_points) > 0:
                self.ax_labeled.scatter(self.tru_points[:, 0], self.tru_points[:, 1], 
                                     self.tru_points[:, 2], s=2, c='red', alpha=0.7, label='Truth')
            self.ax_labeled.legend(loc='upper right')
            
            # Panel 4: Tagged reconstruction data
            # Color reconstruction points based on label: -2 (gray), -1 (green), 0 (blue), 1 (red)
            colors = np.array(['gray', 'green', 'blue', 'red'])[self.labels.astype(int) + 2]
            self.ax_tagged.scatter(self.x, self.y, self.z, s=2, c=colors, alpha=0.7)
            
            # Set labels for all axes
            for ax in [self.ax_rec, self.ax_tru, self.ax_labeled, self.ax_tagged]:
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                
        else:  # 2D views
            if self.view_mode == '2d_xy':
                # Panel 1: Reconstruction data
                self.ax_rec.scatter(self.x, self.y, s=2, c=self.point_color, alpha=0.7)
                
                # Panel 2: Truth data
                tru_colors = np.array(['gray', 'green', 'blue', 'red'])[self.tru_labels.astype(int) + 2]
                if len(self.tru_points) > 0:
                    if len(self.tru_labels) == len(self.tru_points):
                        self.ax_tru.scatter(self.tru_points[:, 0], self.tru_points[:, 1], 
                                            s=2, c=tru_colors, cmap='plasma', alpha=0.7)
                    else:
                        self.ax_tru.scatter(self.tru_points[:, 0], self.tru_points[:, 1], 
                                            s=2, c='red', alpha=0.7)
                
                # Panel 3: Combined truth and reconstruction data
                self.ax_labeled.scatter(self.x, self.y, s=2, c='blue', alpha=0.7, label='Reconstruction')
                if len(self.tru_points) > 0:
                    self.ax_labeled.scatter(self.tru_points[:, 0], self.tru_points[:, 1], 
                                         s=2, c='red', alpha=0.7, label='Truth')
                self.ax_labeled.legend(loc='upper right')
                
                # Panel 4: Tagged reconstruction data
                colors = np.array(['gray', 'green', 'blue', 'red'])[self.labels.astype(int) + 2]
                self.ax_tagged.scatter(self.x, self.y, s=2, c=colors, alpha=0.7)
                
                # Set labels
                for ax in [self.ax_rec, self.ax_tru, self.ax_labeled, self.ax_tagged]:
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    
            elif self.view_mode == '2d_xz':
                # Panel 1: Reconstruction data
                self.ax_rec.scatter(self.x, self.z, s=2, c=self.point_color, alpha=0.7)
                
                # Panel 2: Truth data
                tru_colors = np.array(['gray', 'green', 'blue', 'red'])[self.tru_labels.astype(int) + 2]
                if len(self.tru_points) > 0:
                    if len(self.tru_labels) == len(self.tru_points):
                        self.ax_tru.scatter(self.tru_points[:, 0], self.tru_points[:, 2], 
                                            s=2, c=tru_colors, cmap='plasma', alpha=0.7)
                    else:
                        self.ax_tru.scatter(self.tru_points[:, 0], self.tru_points[:, 2], 
                                            s=2, c='red', alpha=0.7)
                
                # Panel 3: Combined truth and reconstruction data
                self.ax_labeled.scatter(self.x, self.z, s=2, c='blue', alpha=0.7, label='Reconstruction')
                if len(self.tru_points) > 0:
                    self.ax_labeled.scatter(self.tru_points[:, 0], self.tru_points[:, 2], 
                                         s=2, c='red', alpha=0.7, label='Truth')
                self.ax_labeled.legend(loc='upper right')
                
                # Panel 4: Tagged reconstruction data
                colors = np.array(['gray', 'green', 'blue', 'red'])[self.labels.astype(int) + 2]
                self.ax_tagged.scatter(self.x, self.z, s=2, c=colors, alpha=0.7)
                
                # Set labels
                for ax in [self.ax_rec, self.ax_tru, self.ax_labeled, self.ax_tagged]:
                    ax.set_xlabel('X')
                    ax.set_ylabel('Z')
                    
            else:  # '2d_yz'
                # Panel 1: Reconstruction data
                self.ax_rec.scatter(self.y, self.z, s=2, c=self.point_color, alpha=0.7)
                
                # Panel 2: Truth data
                tru_colors = np.array(['gray', 'green', 'blue', 'red'])[self.tru_labels.astype(int) + 2]
                if len(self.tru_points) > 0:
                    if len(self.tru_labels) == len(self.tru_points):
                        self.ax_tru.scatter(self.tru_points[:, 1], self.tru_points[:, 2], 
                                            s=2, c=tru_colors, cmap='plasma', alpha=0.7)
                    else:
                        self.ax_tru.scatter(self.tru_points[:, 1], self.tru_points[:, 2], 
                                            s=2, c='red', alpha=0.7)
                
                # Panel 3: Combined truth and reconstruction data
                self.ax_labeled.scatter(self.y, self.z, s=2, c='blue', alpha=0.7, label='Reconstruction')
                if len(self.tru_points) > 0:
                    self.ax_labeled.scatter(self.tru_points[:, 1], self.tru_points[:, 2], 
                                         s=2, c='red', alpha=0.7, label='Truth')
                self.ax_labeled.legend(loc='upper right')
                
                # Panel 4: Tagged reconstruction data
                colors = np.array(['gray', 'green', 'blue', 'red'])[self.labels.astype(int) + 2]
                self.ax_tagged.scatter(self.y, self.z, s=2, c=colors, alpha=0.7)
                
                # Set labels
                for ax in [self.ax_rec, self.ax_tru, self.ax_labeled, self.ax_tagged]:
                    ax.set_xlabel('Y')
                    ax.set_ylabel('Z')
        
        # Set titles
        self.ax_rec.set_title('Reconstruction')
        self.ax_tru.set_title('Truth')
        self.ax_labeled.set_title('Combined Truth (red) and Reconstruction (blue)')
        self.ax_tagged.set_title('Tagged Reconstruction (by label)')
        
        # Add legend for the tagged reconstruction panel
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='No match (-2)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Non-trackable (-1)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Non-neutrino (0)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Neutrino (1)')
        ]
        self.ax_tagged.legend(handles=legend_elements, loc='upper left')
        
        # Update the figure with tight layout, leaving space at the bottom for status
        plt.tight_layout(rect=[0, 0.05, 0.77, 0.95])  # Adjusted for right control panel
        
        # Update status text
        self.update_status_text()
        
        plt.draw()
    
    def show(self):
        """Show the interactive display"""
        plt.tight_layout(rect=[0, 0.05, 0.77, 0.95])  # Adjusted for right control panel
        plt.show()


def load_rec_file(file_path, z_offset=-25):
    """Load reconstruction data from NPZ file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reconstruction file not found: {file_path}")
    
    try:
        data = np.load(file_path)
        points = data['points']
        points.setflags(write=1)  # Enable writing
        points[:, 2] += z_offset
        print(f"data['points'][0, 2] = {data['points'][0, 2]}")
        data['points'][:, 2] += z_offset
        print(f"data['points'][0, 2] = {data['points'][0, 2]} with z_offset {z_offset} ")
        print(f"Successfully loaded reconstruction file: {file_path}")
        return data
    except Exception as e:
        raise RuntimeError(f"Error loading reconstruction file: {str(e)}")


def load_tru_file(file_path):
    """Load truth data from JSON file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Truth file not found: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded truth file: {file_path}")
        return data
    except Exception as e:
        raise RuntimeError(f"Error loading truth file: {str(e)}")


def load_file_list(file_path):
    """
    Load a list of files from a text file.
    Each line in the file is considered a file path.
    
    Parameters:
    -----------
    file_path : str
        Path to the file containing a list of file paths
        
    Returns:
    --------
    list
        List of file paths
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File list not found: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            # Read lines and remove whitespace, skip empty lines
            file_list = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(file_list)} entries from list file: {file_path}")
        return file_list
    except Exception as e:
        raise RuntimeError(f"Error loading file list: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Check reconstruction and truth data with interactive display')
    parser.add_argument('--rec', required=True, help='Path to file containing a list of reconstruction NPZ files')
    parser.add_argument('--tru', required=True, help='Path to file containing a list of truth JSON files')
    parser.add_argument('--distance-cut', type=float, default=2.0, 
                        help='Initial distance cut for matching points (cm)')
    
    args = parser.parse_args()
    
    try:
        # Load file lists from the provided files
        rec_files = load_file_list(args.rec)
        tru_files = load_file_list(args.tru)
        
        # Ensure that the number of rec and truth files match
        if len(rec_files) != len(tru_files):
            print(f"Error: Number of reconstruction files ({len(rec_files)}) must match number of truth files ({len(tru_files)})")
            sys.exit(1)
        
        if len(rec_files) == 0:
            print("Error: No files found in the provided lists")
            sys.exit(1)
            
        # Create interactive display with lists of files
        display = EnhancedEventDisplay(
            rec_files=rec_files,
            tru_files=tru_files,
            initial_distance_cut=args.distance_cut
        )
        
        # Show the interactive display
        display.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
