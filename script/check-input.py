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
    def __init__(self, rec_data, tru_data, rec_file, tru_file, initial_distance_cut=2.0):
        """
        Interactive event display for visualization of reconstruction, truth, and labeled data.
        
        Parameters:
        -----------
        rec_data : dict
            Reconstruction data loaded from NPZ file
        tru_data : dict
            Truth data loaded from JSON file
        rec_file : str
            Path to reconstruction file
        tru_file : str
            Path to truth file
        initial_distance_cut : float
            Initial value for distance cut in cm for matching points
        """
        self.rec_data = rec_data
        self.tru_data = tru_data
        self.rec_file = rec_file
        self.tru_file = tru_file
        self.distance_cut = initial_distance_cut
        self.view_mode = '3d'  # Can be '3d', '2d_xy', '2d_xz', '2d_yz'
        
        # Extract points from rec_data
        self.points = rec_data['points']
        self.x = self.points[:, 0]
        self.y = self.points[:, 1]
        self.z = self.points[:, 2]
        
        # Apply initial labeling
        self.update_labels()
        
        # Extract truth points and labels for easier access
        self.extract_truth_data()
        
        # Setup the figure and plots
        self.fig = plt.figure(figsize=(18, 8))
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
            self.labels = get_isnu_labels(self.tru_file, self.rec_file, self.distance_cut)
            print(f"Labels updated with distance cut = {self.distance_cut}")
            print(f"Label counts: {np.bincount(self.labels.astype(int) + 2)}")
        except Exception as e:
            print(f"Error updating labels: {str(e)}")
            self.labels = np.zeros(len(self.points))
    
    def setup_plot(self):
        """Set up the plot with panels and interactive controls"""
        # Clear previous figure content
        self.fig.clear()
        
        # Create subplot grid: 3 panels + space for controls
        if self.view_mode == '3d':
            self.ax_rec = self.fig.add_subplot(131, projection='3d')
            self.ax_tru = self.fig.add_subplot(132, projection='3d')
            self.ax_labeled = self.fig.add_subplot(133, projection='3d')
        else:
            self.ax_rec = self.fig.add_subplot(131)
            self.ax_tru = self.fig.add_subplot(132)
            self.ax_labeled = self.fig.add_subplot(133)
        
        # Add control buttons and slider
        self.add_controls()
        
        # Update the display
        self.update_display()
    
    def add_controls(self):
        """Add interactive controls to the figure"""
        # Add buttons for view mode
        ax_3d = plt.axes([0.15, 0.05, 0.05, 0.04])
        ax_xy = plt.axes([0.25, 0.05, 0.05, 0.04])
        ax_xz = plt.axes([0.35, 0.05, 0.05, 0.04])
        ax_yz = plt.axes([0.45, 0.05, 0.05, 0.04])
        
        self.btn_3d = Button(ax_3d, '3D')
        self.btn_xy = Button(ax_xy, 'XY')
        self.btn_xz = Button(ax_xz, 'XZ')
        self.btn_yz = Button(ax_yz, 'YZ')
        
        self.btn_3d.on_clicked(self.view_3d)
        self.btn_xy.on_clicked(self.view_xy)
        self.btn_xz.on_clicked(self.view_xz)
        self.btn_yz.on_clicked(self.view_yz)
        
        # Add slider for distance cut
        ax_slider = plt.axes([0.15, 0.12, 0.65, 0.03])
        self.slider = Slider(
            ax_slider, 'Distance Cut (cm)', 
            0.1, 10.0, 
            valinit=self.distance_cut,
            valstep=0.1
        )
        
        # Add button to update labels
        ax_update = plt.axes([0.7, 0.05, 0.15, 0.04])
        self.btn_update = Button(ax_update, 'Update Labels')
        self.btn_update.on_clicked(self.on_update_labels)
        
        # Connect slider to update function
        self.slider.on_changed(self.on_slider_changed)
    
    def on_slider_changed(self, val):
        """Called when the slider value changes"""
        self.distance_cut = val
        # We don't update labels automatically to avoid expensive computation
        # The user needs to click the 'Update Labels' button
        plt.figtext(0.5, 0.01, f"Current distance cut: {self.distance_cut:.1f} cm (click 'Update Labels' to apply)", 
                   ha='center', fontsize=10)
        plt.draw()
    
    def on_update_labels(self, event):
        """Called when the Update Labels button is clicked"""
        self.update_labels()
        self.update_display()
    
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
        """Update all three panels with current data and view"""
        # Clear axes
        self.ax_rec.clear()
        self.ax_tru.clear()
        self.ax_labeled.clear()
        
        # Plot differently based on view mode
        if self.view_mode == '3d':
            # Panel 1: Reconstruction data
            self.ax_rec.scatter(self.x, self.y, self.z, s=2, alpha=0.7)
            
            # Panel 2: Truth data
            if len(self.tru_points) > 0:
                # Use truth label for coloring if available
                if len(self.tru_labels) == len(self.tru_points):
                    self.ax_tru.scatter(self.tru_points[:, 0], self.tru_points[:, 1], 
                                        self.tru_points[:, 2], s=2, c=self.tru_labels,
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
            
            # Set labels for all axes
            for ax in [self.ax_rec, self.ax_tru, self.ax_labeled]:
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                
        else:  # 2D views
            if self.view_mode == '2d_xy':
                # Panel 1: Reconstruction data
                self.ax_rec.scatter(self.x, self.y, s=2, alpha=0.7)
                
                # Panel 2: Truth data
                if len(self.tru_points) > 0:
                    if len(self.tru_labels) == len(self.tru_points):
                        self.ax_tru.scatter(self.tru_points[:, 0], self.tru_points[:, 1], 
                                            s=2, c=self.tru_labels, cmap='plasma', alpha=0.7)
                    else:
                        self.ax_tru.scatter(self.tru_points[:, 0], self.tru_points[:, 1], 
                                            s=2, c='red', alpha=0.7)
                
                # Panel 3: Combined truth and reconstruction data
                self.ax_labeled.scatter(self.x, self.y, s=2, c='blue', alpha=0.7, label='Reconstruction')
                if len(self.tru_points) > 0:
                    self.ax_labeled.scatter(self.tru_points[:, 0], self.tru_points[:, 1], 
                                         s=2, c='red', alpha=0.7, label='Truth')
                self.ax_labeled.legend(loc='upper right')
                
                # Set labels
                for ax in [self.ax_rec, self.ax_tru, self.ax_labeled]:
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    
            elif self.view_mode == '2d_xz':
                # Panel 1: Reconstruction data
                self.ax_rec.scatter(self.x, self.z, s=2, alpha=0.7)
                
                # Panel 2: Truth data
                if len(self.tru_points) > 0:
                    if len(self.tru_labels) == len(self.tru_points):
                        self.ax_tru.scatter(self.tru_points[:, 0], self.tru_points[:, 2], 
                                            s=2, c=self.tru_labels, cmap='plasma', alpha=0.7)
                    else:
                        self.ax_tru.scatter(self.tru_points[:, 0], self.tru_points[:, 2], 
                                            s=2, c='red', alpha=0.7)
                
                # Panel 3: Combined truth and reconstruction data
                self.ax_labeled.scatter(self.x, self.z, s=2, c='blue', alpha=0.7, label='Reconstruction')
                if len(self.tru_points) > 0:
                    self.ax_labeled.scatter(self.tru_points[:, 0], self.tru_points[:, 2], 
                                         s=2, c='red', alpha=0.7, label='Truth')
                self.ax_labeled.legend(loc='upper right')
                
                # Set labels
                for ax in [self.ax_rec, self.ax_tru, self.ax_labeled]:
                    ax.set_xlabel('X')
                    ax.set_ylabel('Z')
                    
            else:  # '2d_yz'
                # Panel 1: Reconstruction data
                self.ax_rec.scatter(self.y, self.z, s=2, alpha=0.7)
                
                # Panel 2: Truth data
                if len(self.tru_points) > 0:
                    if len(self.tru_labels) == len(self.tru_points):
                        self.ax_tru.scatter(self.tru_points[:, 1], self.tru_points[:, 2], 
                                            s=2, c=self.tru_labels, cmap='plasma', alpha=0.7)
                    else:
                        self.ax_tru.scatter(self.tru_points[:, 1], self.tru_points[:, 2], 
                                            s=2, c='red', alpha=0.7)
                
                # Panel 3: Combined truth and reconstruction data
                self.ax_labeled.scatter(self.y, self.z, s=2, c='blue', alpha=0.7, label='Reconstruction')
                if len(self.tru_points) > 0:
                    self.ax_labeled.scatter(self.tru_points[:, 1], self.tru_points[:, 2], 
                                         s=2, c='red', alpha=0.7, label='Truth')
                self.ax_labeled.legend(loc='upper right')
                
                # Set labels
                for ax in [self.ax_rec, self.ax_tru, self.ax_labeled]:
                    ax.set_xlabel('Y')
                    ax.set_ylabel('Z')
        
        # Set titles
        self.ax_rec.set_title('Reconstruction')
        self.ax_tru.set_title('Truth')
        self.ax_labeled.set_title('Combined Truth (red) and Reconstruction (blue)')
        
        # Update the figure
        plt.tight_layout(rect=[0, 0.15, 1, 0.95])  # Make room for controls
        
        # Display current distance cut
        plt.figtext(0.5, 0.01, f"Current distance cut: {self.distance_cut:.1f} cm", 
                   ha='center', fontsize=10)
        
        # Display view mode
        mode_str = self.view_mode.upper().replace('2D_', '')
        plt.figtext(0.5, 0.025, f"View: {mode_str}", 
                   ha='center', fontsize=10)
        
        plt.draw()
    
    def show(self):
        """Show the interactive display"""
        plt.tight_layout(rect=[0, 0.15, 1, 0.95])  # Make room for controls
        plt.show()


def load_rec_file(file_path):
    """Load reconstruction data from NPZ file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reconstruction file not found: {file_path}")
    
    try:
        data = np.load(file_path)
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


def main():
    parser = argparse.ArgumentParser(description='Check reconstruction and truth data with interactive display')
    parser.add_argument('--rec', required=True, help='Path to reconstruction NPZ file')
    parser.add_argument('--tru', required=True, help='Path to truth JSON file')
    parser.add_argument('--distance-cut', type=float, default=2.0, 
                        help='Initial distance cut for matching points (cm)')
    
    args = parser.parse_args()
    
    try:
        # Load data files
        rec_data = load_rec_file(args.rec)
        tru_data = load_tru_file(args.tru)
        
        # Create interactive display
        display = EnhancedEventDisplay(
            rec_data=rec_data,
            tru_data=tru_data,
            rec_file=args.rec,
            tru_file=args.tru,
            initial_distance_cut=args.distance_cut
        )
        
        # Show the interactive display
        display.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
